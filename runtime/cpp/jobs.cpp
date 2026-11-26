// The fabric's job loop in C++: the register writes, DMA transfers and polling that
// runtime/fabric.py would otherwise do from Python for every AM frame and char-LM step.
//
// PYNQ still loads the overlay and the weights and allocates the two DMA buffers; this
// gets their mappings (the accelerator's and the DMA's register arrays, and the frame and
// score buffers with their physical addresses) and runs batches of jobs on them.
// Registers: shared/specs/fabric_interface.md. DMA: Xilinx AXI DMA in simple mode, the
// same offsets PYNQ's pynq/lib/dma.py uses. Both buffers are allocated non-cacheable
// (fabric.py), so no cache maintenance, only barriers so the CPU's writes land before a
// transfer starts and the DMA's land before they are read.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

// Accelerator registers (byte offsets).
constexpr int CTRL = 0x00, JOB = 0x04, BEAM = 0x08, LM_CHAR = 0x0C, STATUS = 0x10;
constexpr int LOAD_TARGET = 0x14, LOAD_ADDR = 0x18, LOAD_COUNT = 0x1C, SRC_BEAM = 0x28;
constexpr uint32_t DONE = 0x2, ERROR = 0x4;
constexpr uint32_t JOB_AM = 0, JOB_LM = 1, JOB_CLEAR_AM = 2, JOB_CLEAR_BEAM = 3;
constexpr uint32_t T_FRAME = 0;
constexpr int N_IN = 123, N_SCORES = 30, FRAME_BYTES = 124, SCORE_BYTES = 32;

// AXI DMA channel registers, from the channel's base (MM2S 0x00, S2MM 0x30).
constexpr int MM2S = 0x00, S2MM = 0x30;
constexpr int DMA_SR = 0x04, DMA_ADDR = 0x18, DMA_ADDR_MSB = 0x1C, DMA_LENGTH = 0x28;
constexpr uint32_t DMA_IDLE = 0x2, DMA_ERRORS = 0x70;

constexpr auto TIMEOUT = std::chrono::seconds(1);

inline void barrier() { asm volatile("dsb sy" ::: "memory"); }

template <typename T>
T* mapped(const py::buffer& b, size_t min_bytes, const char* what) {
  py::buffer_info info = b.request(true);
  if (static_cast<size_t>(info.size * info.itemsize) < min_bytes)
    throw std::invalid_argument(std::string(what) + " is smaller than expected");
  return static_cast<T*>(info.ptr);
}

class Jobs {
 public:
  Jobs(const py::buffer& regs, const py::buffer& dma, const py::buffer& frame, uint64_t frame_phys,
       const py::buffer& scores, uint64_t scores_phys)
      : regs_(mapped<volatile uint32_t>(regs, 0x2C, "regs")),
        dma_(mapped<volatile uint32_t>(dma, 0x5C, "dma")),
        frame_(mapped<uint8_t>(frame, FRAME_BYTES, "frame")),
        scores_(mapped<int8_t>(scores, SCORE_BYTES, "scores")),
        frame_phys_(frame_phys),
        scores_phys_(scores_phys) {
    std::memset(frame_, 0, FRAME_BYTES);
  }

  // A clear job (AM context, or one char-LM slot).
  void clear(uint32_t job, uint32_t slot) {
    if (job != JOB_CLEAR_AM && job != JOB_CLEAR_BEAM) throw std::invalid_argument("not a clear job");
    start(job, 0, slot, 0);
    wait_done(job);
  }

  // The AM on every frame of q [T, N_IN], from the context as it is: [T, N_SCORES].
  py::array_t<int8_t> am(py::array_t<int8_t, py::array::c_style | py::array::forcecast> q) {
    if (q.ndim() != 2 || q.shape(1) != N_IN) throw std::invalid_argument("q must be [T, 123]");
    const ssize_t n = q.shape(0);
    py::array_t<int8_t> out({n, static_cast<ssize_t>(N_SCORES)});
    const int8_t* in = q.data();
    int8_t* o = out.mutable_data();
    for (ssize_t t = 0; t < n; ++t) {
      load_frame(in + t * N_IN);
      scored(JOB_AM, 0, 0, 0, o + t * N_SCORES);
    }
    return out;
  }

  // Char-LM steps: slot dst[k] = step(slot src[k], chars[k]), in order: [N, N_SCORES].
  py::array_t<int8_t> lm(py::array_t<int32_t, py::array::c_style | py::array::forcecast> src,
                         py::array_t<int32_t, py::array::c_style | py::array::forcecast> dst,
                         py::array_t<int32_t, py::array::c_style | py::array::forcecast> chars) {
    const ssize_t n = src.size();
    if (dst.size() != n || chars.size() != n) throw std::invalid_argument("src, dst, chars differ in length");
    py::array_t<int8_t> out({n, static_cast<ssize_t>(N_SCORES)});
    int8_t* o = out.mutable_data();
    for (ssize_t k = 0; k < n; ++k)
      scored(JOB_LM, static_cast<uint32_t>(src.data()[k]), static_cast<uint32_t>(dst.data()[k]),
             static_cast<uint32_t>(chars.data()[k]), o + k * N_SCORES);
    return out;
  }

 private:
  uint32_t reg(int off) const { return regs_[off / 4]; }
  void set(int off, uint32_t v) { regs_[off / 4] = v; }
  uint32_t dreg(int off) const { return dma_[off / 4]; }
  void dset(int off, uint32_t v) { dma_[off / 4] = v; }

  void transfer(int ch, uint64_t phys, uint32_t nbytes) {
    dset(ch + DMA_ADDR, static_cast<uint32_t>(phys));
    dset(ch + DMA_ADDR_MSB, static_cast<uint32_t>(phys >> 32));
    dset(ch + DMA_LENGTH, nbytes);  // starts the transfer
  }

  void wait_dma(int ch) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
      const uint32_t sr = dreg(ch + DMA_SR);
      if (sr & DMA_ERRORS) throw std::runtime_error("DMA error, status " + std::to_string(sr));
      if (sr & DMA_IDLE) break;
      if (std::chrono::steady_clock::now() - t0 > TIMEOUT)
        throw std::runtime_error("DMA not idle after 1 s, status " + std::to_string(sr));
    }
    barrier();
  }

  void load_frame(const int8_t* q) {
    std::memcpy(frame_, q, N_IN);
    barrier();
    set(LOAD_TARGET, T_FRAME);
    set(LOAD_ADDR, 0);
    transfer(MM2S, frame_phys_, FRAME_BYTES);
    wait_dma(MM2S);
    const uint32_t count = reg(LOAD_COUNT);
    if (count != N_IN) throw std::runtime_error("frame LOAD_COUNT " + std::to_string(count) + ", expected 123");
  }

  void start(uint32_t job, uint32_t src, uint32_t beam, uint32_t ch) {
    set(JOB, job);
    set(SRC_BEAM, src);
    set(BEAM, beam);
    set(LM_CHAR, ch);
    set(CTRL, 1);  // also clears STATUS.done
  }

  void wait_done(uint32_t job) {
    const auto t0 = std::chrono::steady_clock::now();
    uint32_t st;
    while (!((st = reg(STATUS)) & DONE)) {
      if (std::chrono::steady_clock::now() - t0 > TIMEOUT)
        throw std::runtime_error("job " + std::to_string(job) + " not done after 1 s, STATUS " +
                                 std::to_string(st));
    }
    if (st & ERROR) throw std::runtime_error("job " + std::to_string(job) + ": STATUS error bit set");
  }

  // An AM or LM job: arm the score transfer, run the job, copy the scores to out.
  void scored(uint32_t job, uint32_t src, uint32_t beam, uint32_t ch, int8_t* out) {
    transfer(S2MM, scores_phys_, SCORE_BYTES);
    start(job, src, beam, ch);
    wait_done(job);
    wait_dma(S2MM);
    std::memcpy(out, scores_, N_SCORES);
  }

  volatile uint32_t* regs_;
  volatile uint32_t* dma_;
  uint8_t* frame_;
  int8_t* scores_;
  uint64_t frame_phys_, scores_phys_;
};

}  // namespace

PYBIND11_MODULE(_jobs, m) {
  m.doc() = "The fabric's job loop: AM frames and char-LM steps without Python per job.";
  py::class_<Jobs>(m, "Jobs")
      .def(py::init<const py::buffer&, const py::buffer&, const py::buffer&, uint64_t,
                    const py::buffer&, uint64_t>(),
           py::arg("regs"), py::arg("dma"), py::arg("frame"), py::arg("frame_phys"),
           py::arg("scores"), py::arg("scores_phys"))
      .def("clear", &Jobs::clear, py::arg("job"), py::arg("slot") = 0)
      .def("am", &Jobs::am, py::arg("q"))
      .def("lm", &Jobs::lm, py::arg("src"), py::arg("dst"), py::arg("chars"));
}
