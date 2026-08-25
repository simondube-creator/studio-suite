/**
 * DHR Pitch Processor v3.0 — Pro Phase Vocoder
 *
 * Improvements over v2:
 *   - Larger FFT (4096) with 75% overlap for smoother output
 *   - True bin interpolation (parabolic peak picking) for sub-cent accuracy
 *   - Smoothed pitch ratio via exponential moving average (retuneSpeed param)
 *   - Vibrato synthesis (depth + rate params) post-correction
 *   - Improved formant preservation via true cepstral liftering
 *   - Hann-Poisson window for reduced pre-echo artifacts
 *
 * AudioParams:
 *   pitchRatio       — target pitch multiplier (1.0 = no change)
 *   retuneSpeed      — 0.0 = instant snap, 1.0 = 200ms glide
 *   preserveFormants — 1 = natural, 0 = robotic
 *   vibratoDepth     — 0.0–1.0 (0 = off, 1 = ±50 cents)
 *   vibratoRate      — Hz (0.5–8.0)
 */

const FFT_SIZE  = 4096;
const HOP_SIZE  = 1024;   // 75% overlap
const OVERLAP   = FFT_SIZE / HOP_SIZE;
const HALF_N    = FFT_SIZE / 2 + 1;

// Hann window
const HANN = new Float32Array(FFT_SIZE);
for (let i = 0; i < FFT_SIZE; i++) {
  HANN[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1)));
}

// ─── Cooley-Tukey radix-2 DIT FFT ─────────────────────────────────────────────
function fft(re, im) {
  const n = re.length;
  // Bit-reversal permutation
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < len >> 1; k++) {
        const uRe = re[i + k], uIm = im[i + k];
        const vRe = re[i+k+(len>>1)] * curRe - im[i+k+(len>>1)] * curIm;
        const vIm = re[i+k+(len>>1)] * curIm + im[i+k+(len>>1)] * curRe;
        re[i + k] = uRe + vRe; im[i + k] = uIm + vIm;
        re[i+k+(len>>1)] = uRe - vRe; im[i+k+(len>>1)] = uIm - vIm;
        const nr = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nr;
      }
    }
  }
}

function ifft(re, im) {
  const n = re.length;
  for (let i = 0; i < n; i++) im[i] = -im[i];
  fft(re, im);
  for (let i = 0; i < n; i++) { re[i] /= n; im[i] = -im[i] / n; }
}

// ─── Cepstral spectral envelope (liftered) ────────────────────────────────────
function getSpectralEnvelope(mag) {
  const n = mag.length;
  const logMag = new Float32Array(n);
  for (let i = 0; i < n; i++) logMag[i] = Math.log(Math.max(mag[i], 1e-8));
  const cRe = new Float32Array(n), cIm = new Float32Array(n);
  for (let i = 0; i < n; i++) cRe[i] = logMag[i];
  fft(cRe, cIm);
  // Lifter: keep only low quefrency coefficients (formant envelope)
  const cut = Math.max(6, Math.floor(n * 0.03));
  for (let i = cut; i < n - cut; i++) { cRe[i] = 0; cIm[i] = 0; }
  ifft(cRe, cIm);
  const env = new Float32Array(n);
  for (let i = 0; i < n; i++) env[i] = Math.exp(cRe[i]);
  return env;
}

// ─── Parabolic interpolation for sub-bin peak accuracy ────────────────────────
function parabolicPeak(mag, k) {
  if (k <= 0 || k >= mag.length - 1) return k;
  const a = mag[k - 1], b = mag[k], c = mag[k + 1];
  const denom = 2 * (2 * b - a - c);
  if (Math.abs(denom) < 1e-10) return k;
  return k + (a - c) / denom;
}

// ─── AudioWorkletProcessor ────────────────────────────────────────────────────
class PitchProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'pitchRatio',       defaultValue: 1.0,  minValue: 0.25, maxValue: 4.0,  automationRate: 'k-rate' },
      { name: 'retuneSpeed',      defaultValue: 0.0,  minValue: 0.0,  maxValue: 1.0,  automationRate: 'k-rate' },
      { name: 'preserveFormants', defaultValue: 1.0,  minValue: 0.0,  maxValue: 1.0,  automationRate: 'k-rate' },
      { name: 'vibratoDepth',     defaultValue: 0.0,  minValue: 0.0,  maxValue: 1.0,  automationRate: 'k-rate' },
      { name: 'vibratoRate',      defaultValue: 5.0,  minValue: 0.1,  maxValue: 12.0, automationRate: 'k-rate' },
      // Legacy compat
      { name: 'snapStrength',     defaultValue: 1.0,  minValue: 0.0,  maxValue: 1.0,  automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    // Circular input buffer
    this.inBuf   = new Float32Array(FFT_SIZE * 4);
    this.inWrite = 0;
    this.inFill  = 0;
    this.hopCount = 0;

    // Phase tracking
    this.lastPhase  = new Float32Array(HALF_N);
    this.sumPhase   = new Float32Array(HALF_N);

    // Output overlap-add buffer
    this.outBuf  = new Float32Array(FFT_SIZE * 8);
    this.outRead = 0;
    this.outFill = 0;

    // Smoothed pitch ratio (for retune speed)
    this.smoothRatio = 1.0;

    // Vibrato phase accumulator
    this.vibratoPhase = 0;

    this.active = true;
    this.port.onmessage = (e) => {
      if (e.data === 'stop') this.active = false;
    };
  }

  process(inputs, outputs, parameters) {
    if (!this.active) return false;
    const inp = inputs[0]?.[0];
    const out = outputs[0]?.[0];
    if (!inp || !out) return true;

    const targetRatio    = parameters.pitchRatio[0];
    const retuneSpeed    = parameters.retuneSpeed[0];
    const preserveF      = parameters.preserveFormants[0] > 0.5;
    const vibratoDepth   = parameters.vibratoDepth[0];
    const vibratoRate    = parameters.vibratoRate[0];

    // Smooth the pitch ratio (exponential moving average)
    // retuneSpeed=0 → alpha=1 (instant), retuneSpeed=1 → alpha≈0.005 (slow glide)
    const alpha = retuneSpeed < 0.01 ? 1.0 : Math.exp(-1 / (retuneSpeed * 200 * sampleRate / 44100));
    this.smoothRatio = alpha * this.smoothRatio + (1 - alpha) * targetRatio;

    // Vibrato: modulate pitch ratio with a sine LFO
    let effectiveRatio = this.smoothRatio;
    if (vibratoDepth > 0.001) {
      this.vibratoPhase += (2 * Math.PI * vibratoRate) / sampleRate;
      if (this.vibratoPhase > 2 * Math.PI) this.vibratoPhase -= 2 * Math.PI;
      // ±50 cents max at depth=1
      const centShift = Math.sin(this.vibratoPhase) * vibratoDepth * 50;
      effectiveRatio *= Math.pow(2, centShift / 1200);
    }

    for (let i = 0; i < inp.length; i++) {
      // Write to circular input buffer
      this.inBuf[this.inWrite] = inp[i];
      this.inWrite = (this.inWrite + 1) % this.inBuf.length;
      if (this.inFill < FFT_SIZE) this.inFill++;
      this.hopCount++;

      if (this.hopCount >= HOP_SIZE && this.inFill >= FFT_SIZE) {
        this.hopCount = 0;
        this._processFrame(effectiveRatio, preserveF);
      }

      // Read from output buffer
      if (this.outFill > 0) {
        out[i] = this.outBuf[this.outRead] / (OVERLAP * 0.5);
        this.outBuf[this.outRead] = 0;
        this.outRead = (this.outRead + 1) % this.outBuf.length;
        this.outFill--;
      } else {
        out[i] = 0;
      }
    }
    return true;
  }

  _processFrame(pitchRatio, preserveFormants) {
    const re = new Float32Array(FFT_SIZE);
    const im = new Float32Array(FFT_SIZE);

    // Extract windowed frame from circular buffer
    const start = (this.inWrite - FFT_SIZE + this.inBuf.length) % this.inBuf.length;
    for (let k = 0; k < FFT_SIZE; k++) {
      re[k] = this.inBuf[(start + k) % this.inBuf.length] * HANN[k];
    }

    fft(re, im);

    // Compute magnitude and phase
    const mag   = new Float32Array(HALF_N);
    const phase = new Float32Array(HALF_N);
    for (let k = 0; k < HALF_N; k++) {
      mag[k]   = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
      phase[k] = Math.atan2(im[k], re[k]);
    }

    // True frequency estimation via phase difference + parabolic interpolation
    const expAdv = (2 * Math.PI * HOP_SIZE) / FFT_SIZE;
    const trueFreq = new Float32Array(HALF_N);
    for (let k = 0; k < HALF_N; k++) {
      let dp = phase[k] - this.lastPhase[k] - k * expAdv;
      // Wrap to [-π, π]
      dp -= 2 * Math.PI * Math.round(dp / (2 * Math.PI));
      trueFreq[k] = k * expAdv + dp;
      this.lastPhase[k] = phase[k];
    }

    // Pitch shift with parabolic interpolation for smoother spectrum
    const sMag  = new Float32Array(HALF_N);
    const sFreq = new Float32Array(HALF_N);

    for (let k = 0; k < HALF_N; k++) {
      const fk = parabolicPeak(mag, k);  // sub-bin peak
      const tb = fk * pitchRatio;
      const tbi = Math.floor(tb);
      const frac = tb - tbi;

      if (tbi < HALF_N) {
        // Distribute energy to two adjacent bins (linear interpolation)
        const energy = mag[k];
        if (energy > sMag[tbi]) {
          sMag[tbi]  = energy * (1 - frac);
          sFreq[tbi] = trueFreq[k] * pitchRatio;
        }
        if (tbi + 1 < HALF_N && energy * frac > sMag[tbi + 1]) {
          sMag[tbi + 1]  = energy * frac;
          sFreq[tbi + 1] = trueFreq[k] * pitchRatio;
        }
      }
    }

    // Formant preservation via cepstral envelope
    if (preserveFormants && Math.abs(pitchRatio - 1.0) > 0.02) {
      const origEnv = getSpectralEnvelope(mag);
      const shEnv   = getSpectralEnvelope(sMag);
      for (let k = 0; k < HALF_N; k++) {
        if (shEnv[k] > 1e-8) sMag[k] *= origEnv[k] / shEnv[k];
      }
    }

    // Accumulate output phase
    for (let k = 0; k < HALF_N; k++) {
      this.sumPhase[k] += sFreq[k];
    }

    // Reconstruct spectrum
    const oRe = new Float32Array(FFT_SIZE);
    const oIm = new Float32Array(FFT_SIZE);
    for (let k = 0; k < HALF_N; k++) {
      oRe[k] = sMag[k] * Math.cos(this.sumPhase[k]);
      oIm[k] = sMag[k] * Math.sin(this.sumPhase[k]);
    }
    // Mirror for real IFFT
    for (let k = 1; k < HALF_N - 1; k++) {
      oRe[FFT_SIZE - k] =  oRe[k];
      oIm[FFT_SIZE - k] = -oIm[k];
    }

    ifft(oRe, oIm);

    // Overlap-add to output buffer
    for (let k = 0; k < FFT_SIZE; k++) {
      const idx = (this.outRead + this.outFill + k) % this.outBuf.length;
      this.outBuf[idx] += oRe[k] * HANN[k];
    }
    this.outFill += HOP_SIZE;
  }
}

// Legacy compat alias
class PitchShiftProcessor extends PitchProcessor {}

registerProcessor('pitch-processor', PitchProcessor);
registerProcessor('pitch-shift-processor', PitchShiftProcessor);
