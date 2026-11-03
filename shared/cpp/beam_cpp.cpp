// C++ port of the beam search in src/shared/decode/beam.py and the KenLM adapter in
// decode/wlm.py, built as the module `shared.decode.beam_cpp` (build.sh next to this file).
// The Python search stays the readable reference and writes the golden case; this is the one
// training and the board run, about 7x faster on the board's A53s.
//
// Same algorithm and the same order of floating-point operations as the Python search (built
// with -ffp-contract=off so no multiply-add is fused), so it returns the same N-best with the
// same scores: tests/test_beam_cpp.py holds it to golden/decode/. KenLM is compiled in and
// called directly. The CM stays a Python object behind the CharLM interface (start / step),
// called once per frame for the new prefixes, as the fabric driver will be.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lm/model.hh"
#include "util/file_piece.hh"

namespace py = pybind11;

namespace {

constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
// shared/vocab.py: blank, A-Z, space, apostrophe, EOS.
constexpr int BLANK_IDX = 0;
constexpr int EOS_IDX = 29;
constexpr int FIRST_CHAR = 1, LAST_CHAR = 28;  // what a prefix can be extended by
constexpr char CHARS[] = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ '";
constexpr char SPACE = ' ';
constexpr double OOV_PENALTY = -10.0;  // log10, as in wlm.py
const double LN10 = std::log(10.0);

int char_index(char ch) { return ch == SPACE ? 27 : ch == '\'' ? 28 : ch - 'A' + 1; }

// The words of an ARPA file (its 1-grams, without <s>, </s>, <unk>), sorted: wlm._unigrams.
std::vector<std::string> unigrams(const std::string& arpa) {
  util::FilePiece f(arpa.c_str());
  StringPiece line;
  while (f.ReadLineOrEOF(line) &&
         std::string_view(line.data(), line.size()).rfind("\\1-grams:", 0) != 0) {
  }
  std::vector<std::string> words;
  while (f.ReadLineOrEOF(line)) {
    std::istringstream fields(std::string(line.data(), line.size()));
    std::string logp, word;
    if (!(fields >> logp)) break;
    fields >> word;
    if (word[0] != '<') words.push_back(word);
  }
  std::sort(words.begin(), words.end());
  return words;
}

// wlm.KenLM: an ARPA model scoring one word at a time, in natural log.
class WordLM {
 public:
  explicit WordLM(const std::string& arpa)
      : model_(lm::ngram::LoadVirtual(arpa.c_str())), words_(unigrams(arpa)) {}

  lm::ngram::State start() const {
    lm::ngram::State s;
    model_->BeginSentenceWrite(&s);
    return s;
  }

  // Natural-log prob of `word` after `in`; `out` gets the state after it.
  double score(const lm::ngram::State& in, const std::string& word, lm::ngram::State* out) const {
    lm::WordIndex w = model_->BaseVocabulary().Index(word);
    double logp = model_->BaseScore(&in, w, out);
    if (w == 0) logp += OOV_PENALTY;
    return logp * LN10;
  }

  double end(const lm::ngram::State& in) const {
    lm::ngram::State out;
    return model_->BaseScore(&in, model_->BaseVocabulary().Index("</s>"), &out) * LN10;
  }

  bool is_prefix(const std::string& text) const {
    auto it = std::lower_bound(words_.begin(), words_.end(), text);
    return it != words_.end() && it->compare(0, text.size(), text) == 0;
  }

  double oov_penalty() const { return OOV_PENALTY * LN10; }

 private:
  std::unique_ptr<lm::base::Model> model_;
  std::vector<std::string> words_;
};

struct Config {
  size_t beam;
  double alpha, beta, gamma, prune;
};

struct Prefix {
  std::string text;
  double lm = 0.0;
  size_t word_start = 0;
  lm::ngram::State wstate;
  py::object cstate;
  std::vector<double> clogp;  // empty until the CM has stepped this prefix
  std::shared_ptr<Prefix> parent;
  int ch = 0;
};
using PrefixPtr = std::shared_ptr<Prefix>;

// A beam (or the next one being built): entries in insertion order, plus text -> position.
// The keys view the prefixes' own text, which the entries keep alive.
struct Entry {
  double p_b, p_nb;
  PrefixPtr pre;
};
struct Beam {
  std::vector<Entry> entries;
  std::unordered_map<std::string_view, size_t> index;

  Entry* find(std::string_view text) {
    auto it = index.find(text);
    return it == index.end() ? nullptr : &entries[it->second];
  }
  Entry& add(PrefixPtr pre) {
    entries.push_back({NEG_INF, NEG_INF, std::move(pre)});
    index.emplace(entries.back().pre->text, entries.size() - 1);
    return entries.back();
  }
};

double logadd(double a, double b) {
  if (a < b) std::swap(a, b);
  if (b == NEG_INF) return a;
  return a + std::log1p(std::exp(b - a));
}

std::vector<double> as_row(const py::handle& logp) {
  auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(logp);
  return std::vector<double>(a.data(), a.data() + a.size());
}

PrefixPtr extend(const PrefixPtr& pre, int c, const Config& cfg, const WordLM* wlm) {
  char ch = CHARS[c];
  auto out = std::make_shared<Prefix>();
  out->text = pre->text + ch;
  double lm = pre->lm;
  if (!pre->clogp.empty()) lm += cfg.alpha * pre->clogp[c];
  out->wstate = pre->wstate;
  out->word_start = pre->word_start;
  if (ch == SPACE) {
    lm += cfg.gamma;
    if (wlm) {
      double logp = wlm->score(pre->wstate, pre->text.substr(pre->word_start), &out->wstate);
      lm += cfg.beta * logp;
    }
    out->word_start = pre->text.size() + 1;
  } else if (wlm && !wlm->is_prefix(out->text.substr(pre->word_start))) {
    lm += cfg.beta * wlm->oov_penalty();
  }
  out->lm = lm;
  if (!pre->clogp.empty()) out->parent = pre;
  out->ch = c;
  return out;
}

void advance_cm(const py::object& cm, const Beam& beam) {
  std::vector<Prefix*> todo;
  for (const Entry& e : beam.entries)
    if (e.pre->clogp.empty()) todo.push_back(e.pre.get());
  if (todo.empty()) return;
  py::list states, chars;
  for (Prefix* p : todo) {
    states.append(p->parent->cstate);
    chars.append(p->ch);
  }
  py::tuple res = cm.attr("step")(states, chars);
  py::list new_states = res[0];
  auto logps = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(res[1]);
  size_t v = logps.shape(1);
  for (size_t n = 0; n < todo.size(); ++n) {
    todo[n]->cstate = new_states[n];
    todo[n]->clogp.assign(logps.data() + n * v, logps.data() + (n + 1) * v);
    todo[n]->parent.reset();
  }
}

double finish(const Prefix& pre, const Config& cfg, const WordLM* wlm) {
  double lm = 0.0;
  if (!pre.clogp.empty()) lm += cfg.alpha * pre.clogp[EOS_IDX];
  std::string word = pre.text.substr(pre.word_start);
  if (!word.empty()) lm += cfg.gamma;
  if (wlm) {
    lm::ngram::State state = pre.wstate, out;
    if (!word.empty()) {
      double logp = wlm->score(state, word, &out);
      state = out;
      lm += cfg.beta * logp;
    }
    lm += cfg.beta * wlm->end(state);
  }
  return lm;
}

std::vector<std::pair<std::string, double>> beam_search(
    py::array_t<double, py::array::c_style | py::array::forcecast> logp, const py::object& py_cfg,
    const py::object& cm, const WordLM* wlm) {
  Config cfg{py_cfg.attr("beam").cast<size_t>(), py_cfg.attr("alpha").cast<double>(),
             py_cfg.attr("beta").cast<double>(), py_cfg.attr("gamma").cast<double>(),
             py_cfg.attr("prune").cast<double>()};
  bool use_cm = !cm.is_none();
  auto root = std::make_shared<Prefix>();
  if (wlm) root->wstate = wlm->start();
  if (use_cm) {
    py::tuple s = cm.attr("start")();
    root->cstate = s[0];
    root->clogp = as_row(s[1]);
  }
  Beam beam;
  beam.add(root).p_b = 0.0;

  size_t n_frames = logp.shape(0), v = logp.shape(1);
  std::vector<int> chars;
  std::vector<size_t> order;
  std::vector<double> keys;
  for (size_t t = 0; t < n_frames; ++t) {
    const double* row = logp.data() + t * v;
    double blank = row[BLANK_IDX];
    chars.clear();
    for (int c = FIRST_CHAR; c <= LAST_CHAR; ++c)
      if (row[c] >= cfg.prune) chars.push_back(c);
    Beam nxt;
    nxt.entries.reserve(beam.entries.size() * (chars.size() + 1));
    for (const Entry& e : beam.entries) {
      const std::string& text = e.pre->text;
      double p = logadd(e.p_b, e.p_nb);
      // Blank, or the last character again: the prefix stays as it is.
      Entry* entry = nxt.find(text);
      if (!entry) entry = &nxt.add(e.pre);
      entry->p_b = logadd(entry->p_b, p + blank);
      if (!text.empty()) entry->p_nb = logadd(entry->p_nb, e.p_nb + row[char_index(text.back())]);
      // A new character. Repeating the last one needs a blank in between.
      for (int c : chars) {
        char ch = CHARS[c];
        if (ch == SPACE && (text.empty() || text.back() == SPACE)) continue;
        std::string next = text + ch;
        entry = nxt.find(next);
        if (!entry) {
          Entry* old = beam.find(next);
          entry = &nxt.add(old ? old->pre : extend(e.pre, c, cfg, wlm));
        }
        double from_p = !text.empty() && text.back() == ch ? e.p_b : p;
        entry->p_nb = logadd(entry->p_nb, from_p + row[c]);
      }
    }
    // heapq.nlargest: the top `beam` by score, ties in insertion order.
    keys.resize(nxt.entries.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      const Entry& e = nxt.entries[i];
      keys[i] = logadd(e.p_b, e.p_nb) + e.pre->lm;
    }
    order.resize(keys.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return keys[a] > keys[b]; });
    order.resize(std::min(order.size(), cfg.beam));
    Beam kept;
    kept.entries.reserve(order.size());
    for (size_t i : order) kept.add(nxt.entries[i].pre) = nxt.entries[i];
    beam = std::move(kept);
    if (use_cm) advance_cm(cm, beam);
  }

  // A prefix with a trailing space ends as the same transcript as the one without: keep the
  // better of the two.
  std::vector<std::pair<std::string, double>> final_;
  std::unordered_map<std::string, size_t> where;
  for (const Entry& e : beam.entries) {
    double score = logadd(e.p_b, e.p_nb) + e.pre->lm + finish(*e.pre, cfg, wlm);
    std::string text = e.pre->text;
    text.erase(text.find_last_not_of(SPACE) + 1);
    auto it = where.find(text);
    double best = it == where.end() ? NEG_INF : final_[it->second].second;
    if (score > best) {
      if (it == where.end()) {
        where.emplace(text, final_.size());
        final_.emplace_back(text, score);
      } else {
        final_[it->second].second = score;
      }
    }
  }
  std::stable_sort(final_.begin(), final_.end(),
                   [](const auto& a, const auto& b) { return a.second > b.second; });
  return final_;
}

}  // namespace

PYBIND11_MODULE(beam_cpp, m) {
  m.doc() = "C++ port of shared.decode.beam.beam_search and the KenLM adapter in wlm.py.";
  py::class_<WordLM>(m, "WordLM")
      .def(py::init<const std::string&>(), py::arg("arpa"));
  m.def("beam_search", &beam_search, py::arg("logp"), py::arg("cfg"),
        py::arg("cm") = py::none(), py::arg("wlm") = py::none());
}
