// juicy_bank_tilde.c — Juicy's modal bank (single object hosting N resonators)
// Patch: 2025-09-03 (stability pass)
//   • FIX gain: "gain" is now a clean 0..1 scalar per mode (independent of body weights).
//     Body weights (position/anisotropy/brightness) are applied on the *input* only, normalized per block.
//   • FIX coupling: smaller, freq‑aware velocity rotation (energy‑preserving), no limiter "drive".
//     Added only a last‑resort scaler if absolute peak > 1.2 to prevent blasts (inaudible when not needed).
//   • Contact: back to original "body" style — soft clip applied to the *exciter* (input), not the resonator state.
//   • Index loudness: kept position floor & gentle compensation so even/odd don’t vanish at p=0.5.
//
// Build: standard Pd external, class name: juicy_bank~

#include "m_pd.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAXN 128

static t_class *juicy_bank_tilde_class;

static inline float  clampf(float x, float lo, float hi){ return (x<lo)?lo:((x>hi)?hi:x); }
static inline double denorm_fix(double v){ return (v<1e-30 && v>-1e-30)?0.0:v; }

typedef struct {
    // user params
    float ratio, gain, attack_ms, decay_ms, pan, keytrack;
    float disp_rand; // per-mode random for dispersion

    // biquad state
    double a1, a2, y1, y2, env, freq_hz, rcur;

    // pan gains
    double gl, gr;

    // per-block weights (we reuse amp_w as normalized input weight)
    float  amp_w, r_w, pos_w, aniso_w, bright_w;

    // misc
    double cstate;

    unsigned char active, dirty_coeffs;
} t_mode;

typedef struct _juicy_bank_tilde {
    t_object  x_obj;
    t_float   f_dummy;

    // global/body
    int     N;
    float   base_hz;
    float   damping, brightness, position, dispersion, coupling, density, anisotropy, contact;
    int     aniso_P;
    float   contact_soft, couple_df;

    // derived
    float   r_mul_base, bright_slope, aniso_gamma;
    float   mix_scale;
    int     n_active;

    // edit cursor
    int     edit_idx;

    // modes
    t_mode  m[MAXN];

    // dsp
    float   sr;
    int     debug;

    // outlets
    t_outlet *outL, *outR, *outBody, *outRes;
} t_juicy_bank_tilde;

// RNG helpers
static unsigned int wang_hash(unsigned int x){ x = (x ^ 61) ^ (x >> 16); x *= 9; x = x ^ (x >> 4); x *= 0x27d4eb2d; x = x ^ (x >> 15); return x; }
static float rand_uni_pm1(unsigned int seed){ unsigned int h = wang_hash(seed); return ((h >> 8) & 0xFFFF) / 32767.5f - 1.0f; }

static void mode_update_pan(t_mode *m){
    double p = clampf(m->pan, -1.f, 1.f);
    double th = (p + 1.0) * 0.25 * M_PI; // -1..+1 -> 0..pi/2
    m->gl = cos(th);
    m->gr = sin(th);
}

static void bank_recalc_body(t_juicy_bank_tilde *x){
    x->damping    = clampf(x->damping,    0.f, 1.f);
    x->brightness = clampf(x->brightness, -1.f, 1.f);
    x->position   = clampf(x->position,   0.f, 1.f);
    x->dispersion = clampf(x->dispersion, -1.f, 1.f);
    x->coupling   = clampf(x->coupling,   0.f, 1.f);
    x->density    = clampf(x->density,   -1.f, 1.f);
    x->anisotropy = clampf(x->anisotropy, -1.f, 1.f);
    x->contact    = clampf(x->contact,    0.f, 1.f);
    if (x->base_hz <= 0) x->base_hz = 440.f;
    if (x->N < 1) x->N = 1;
    if (x->N > MAXN) x->N = MAXN;
    if (x->aniso_P < 1) x->aniso_P = 2;
    if (x->contact_soft < 0.5f) x->contact_soft = 2.f;
    if (x->couple_df <= 0) x->couple_df = 200.f;

    // Damping map: 0..1 -> base pole multiplier (global tilt)
    float d_s = x->damping * x->damping;
    x->r_mul_base = 1.f - 0.20f * d_s; // 1 → 0.8 at full damping
    x->aniso_gamma  = x->anisotropy;
    x->bright_slope = 0.6f * x->brightness;
}

static void bank_emit_body_state(t_juicy_bank_tilde *x){
    t_atom av[64]; int i=0;
    SETSYMBOL(av+i, gensym("base_hz")); i++; SETFLOAT(av+i, x->base_hz); i++;
    SETSYMBOL(av+i, gensym("n_modes")); i++; SETFLOAT(av+i, (t_float)x->N); i++;
    SETSYMBOL(av+i, gensym("n_active")); i++; SETFLOAT(av+i, (t_float)x->n_active); i++;
    SETSYMBOL(av+i, gensym("mix_scale")); i++; SETFLOAT(av+i, x->mix_scale); i++;
    SETSYMBOL(av+i, gensym("damping")); i++; SETFLOAT(av+i, x->damping); i++;
    SETSYMBOL(av+i, gensym("brightness")); i++; SETFLOAT(av+i, x->brightness); i++;
    SETSYMBOL(av+i, gensym("position")); i++; SETFLOAT(av+i, x->position); i++;
    SETSYMBOL(av+i, gensym("dispersion")); i++; SETFLOAT(av+i, x->dispersion); i++;
    SETSYMBOL(av+i, gensym("coupling")); i++; SETFLOAT(av+i, x->coupling); i++;
    SETSYMBOL(av+i, gensym("density")); i++; SETFLOAT(av+i, x->density); i++;
    SETSYMBOL(av+i, gensym("anisotropy")); i++; SETFLOAT(av+i, x->anisotropy); i++;
    SETSYMBOL(av+i, gensym("aniso_P")); i++; SETFLOAT(av+i, (t_float)x->aniso_P); i++;
    SETSYMBOL(av+i, gensym("contact")); i++; SETFLOAT(av+i, x->contact); i++;
    SETSYMBOL(av+i, gensym("contact_soft")); i++; SETFLOAT(av+i, x->contact_soft); i++;
    outlet_anything(x->outBody, gensym("body_state"), i, av);
}

static void bank_emit_res_selected(t_juicy_bank_tilde *x){
    int k = x->edit_idx; if (k<1 || k>x->N) return;
    t_mode *m = &x->m[k-1];
    t_atom av[48]; int i=0;
    SETSYMBOL(av+i, gensym("id")); i++; SETFLOAT(av+i, (t_float)k); i++;
    SETSYMBOL(av+i, gensym("active")); i++; SETFLOAT(av+i, (t_float)(m->active!=0)); i++;
    SETSYMBOL(av+i, gensym("freq")); i++; SETFLOAT(av+i, (t_float)m->freq_hz); i++;
    SETSYMBOL(av+i, gensym("ratio")); i++; SETFLOAT(av+i, (t_float)m->ratio); i++;
    SETSYMBOL(av+i, gensym("key")); i++; SETFLOAT(av+i, (t_float)(m->keytrack!=0)); i++;
    SETSYMBOL(av+i, gensym("gain")); i++; SETFLOAT(av+i, (t_float)m->gain); i++;
    SETSYMBOL(av+i, gensym("attack_ms")); i++; SETFLOAT(av+i, (t_float)m->attack_ms); i++;
    SETSYMBOL(av+i, gensym("decay_ms")); i++; SETFLOAT(av+i, (t_float)m->decay_ms); i++;
    SETSYMBOL(av+i, gensym("pan")); i++; SETFLOAT(av+i, (t_float)m->pan); i++;
    SETSYMBOL(av+i, gensym("pos_w")); i++; SETFLOAT(av+i, m->pos_w); i++;
    SETSYMBOL(av+i, gensym("aniso_w")); i++; SETFLOAT(av+i, m->aniso_w); i++;
    SETSYMBOL(av+i, gensym("bright_w")); i++; SETFLOAT(av+i, m->bright_w); i++;
    outlet_anything(x->outRes, gensym("res_state"), i, av);
}

static void bank_update_coeffs_one(t_juicy_bank_tilde *x, int k){
    if (k < 1 || k > x->N) return;
    t_mode *m = &x->m[k-1];

    double base = (x->base_hz>0?x->base_hz:1.0);

    // Dispersion: unique per mode via disp_rand, knob sets *max spread* (±0.9*|knob|); skip fundamental
    float disp_amt = 0.9f * fabsf(x->dispersion);
    double ratio_eff = (double)m->ratio;
    if (m->keytrack!=0.f){
        if (k != 1){
            ratio_eff = (double)m->ratio + (double)disp_amt * (double)m->disp_rand;
            if (ratio_eff <= 0.0001) ratio_eff = 0.0001;
        } else {
            ratio_eff = (double)m->ratio;
        }
    }
    double fr = (m->keytrack!=0.f) ? base * ratio_eff : (double)m->ratio;
    if (m->keytrack==0.f){
        if (k != 1){
            fr = (double)m->ratio * (1.0 + (double)disp_amt * (double)m->disp_rand);
        } else {
            fr = (double)m->ratio;
        }
        if (fr < 1.0) fr = 1.0;
    }

    // Density: cluster/spread around base (or ratio=1)
    float dens = x->density; // -1..1
    if (m->keytrack!=0.f){
        double d = ratio_eff - 1.0;
        fr = base * (1.0 + d * (1.0 + (double)dens));
    } else {
        double d = fr - base;
        fr = base + d * (1.0 + (double)dens);
    }

    double fs = (double)x->sr;
    if (fr < 1.0) fr = 1.0;
    if (fr > 0.45*fs) fr = 0.45*fs;
    m->freq_hz = fr;

    // Damping & biquad
    double d_s = (m->decay_ms<=0?0.0:m->decay_ms*0.001);
    double r = (d_s<=0.0)? 0.0 : exp(-1.0/(d_s*fs));
    double frel = fr / base; if (frel < 0.1) frel = 0.1;
    double expo = 1.0 + 0.5 * sqrt(frel);
    double r_w = pow((double)x->r_mul_base, expo);
    r *= r_w;
    if (r>0.999999) r = 0.999999;
    if (r<0.0)      r = 0.0;
    m->rcur = r;

    double w = 2.0 * M_PI * fr / fs;
    m->a1 = 2.0 * r * cos(w);
    m->a2 = - (r * r);

    // Position weight (with a floor so even/odd don’t vanish at middle)
    int    idx1 = k;
    double posw = fabs(sin(M_PI * (double)idx1 * (double)clampf(x->position,0.f,1.f)));
    if (posw < 0.20) posw = 0.20;

    // Anisotropy: attenuation-only (period P)
    int P = (x->aniso_P<1?1:x->aniso_P);
    double cyc = cos(2.0*M_PI * (double)(idx1-1) / (double)P);
    double a = x->aniso_gamma;
    double aniso = 1.0;
    if (a > 0){
        if (cyc < 0) aniso = 1.0 - (double)fabs(a) * 0.8;
    } else if (a < 0){
        if (cyc > 0) aniso = 1.0 - (double)fabs(a) * 0.8;
    }

    // Brightness (tempered)
    double bright = pow(fr/base, (double)x->bright_slope);
    if (bright < 0.05) bright = 0.05;
    if (bright > 20.0) bright = 20.0;

    m->pos_w    = (float)posw;
    m->aniso_w  = (float)aniso;
    m->bright_w = (float)bright;
    m->dirty_coeffs = 0;

    mode_update_pan(m);
}

// === DSP ===
static inline double softsat(double x, double soft){ if (soft<=1e-9) soft=1e-9; return tanh(x/soft)*soft; }

static t_int *juicy_bank_tilde_perform(t_int *w){
    t_juicy_bank_tilde *x = (t_juicy_bank_tilde *)(w[1]);
    t_sample *inL = (t_sample *)(w[2]);
    t_sample *inR = (t_sample *)(w[3]);
    t_sample *outL= (t_sample *)(w[4]);
    t_sample *outR= (t_sample *)(w[5]);
    int n = (int)(w[6]);

    bank_recalc_body(x);

    int N = x->N;
    int nact = 0;
    double sumg2 = 0.0;

    // --- compute input-shaping weight, normalized across active modes ---
    double wsum = 0.0;
    for (int k=1; k<=N; ++k){
        t_mode *m = &x->m[k-1];
        m->active = (m->gain > 0.0005f) || (fabs(m->env) > 1e-6);
        bank_update_coeffs_one(x, k);
        if (m->active) {
            nact++;
            sumg2 += (double)m->gain * (double)m->gain;
            double w_in = (double)m->pos_w * (double)m->aniso_w * (double)m->bright_w;
            if (w_in < 0.0) w_in = 0.0;
            if (w_in > 64.0) w_in = 64.0;
            m->amp_w = (float)w_in;  // reuse field to store normalized input weight later
            wsum += w_in;
        } else {
            m->amp_w = 0.f;
        }
    }
    x->n_active = nact;
    x->mix_scale = (sumg2>1e-12)? (float)(1.0 / sqrt(sumg2)) : ((nact>0)? (1.0f / sqrtf((float)nact)) : 1.0f);
    double wnorm = (wsum>0.0)? (wsum / (double)(nact>0?nact:1)) : 1.0;
    if (wnorm <= 0.0) wnorm = 1.0;

    for (int i=0;i<n;++i){ outL[i]=0; outR[i]=0; }

    const double c_amt  = clampf(x->contact, 0.f, 1.f);
    const double c_soft = x->contact_soft;
    double coup = clampf(x->coupling, 0.f, 1.f);

    // coupling angle (very conservative) with easing
    const double theta_max = 0.03; // ~1.7 degrees at full
    const double theta_eased = theta_max * (0.5 - 0.5*cos(M_PI*coup));

    for (int i=0;i<n;++i){
        double xin_mono = 0.5*((double)inL[i] + (double)inR[i]);

        // Contact on EXCITER (original flavor)
        double xin_contact = (c_amt>1e-6)
            ? ((1.0 - c_amt) * xin_mono + c_amt * softsat(xin_mono, c_soft))
            : xin_mono;

        double y1_prev[MAXN];
        double y2_prev[MAXN];
        double y0_buf[MAXN];
        double v_buf[MAXN];
        unsigned char active_mask[MAXN];

        for (int k=1; k<=N; ++k){ 
            t_mode *m = &x->m[k-1];
            y1_prev[k-1]=m->y1; 
            y2_prev[k-1]=m->y2;
            active_mask[k-1]=m->active;
        }

        // ---- pass 1: compute raw y0 (resonator core), env update
        for (int k=1; k<=N; ++k){
            t_mode *m = &x->m[k-1];
            double y1=y1_prev[k-1], y2=y2_prev[k-1];
            if (!active_mask[k-1]){ y0_buf[k-1]=y1; continue; }

            double fs = (double)x->sr;
            double att = (m->attack_ms<=0? 0.0 : exp(-1.0/( (double)m->attack_ms*0.001 * fs )));
            double dec = (m->decay_ms <=0? 0.0 : exp(-1.0/( (double)m->decay_ms *0.001 * fs )));
            double w_in_norm = (double)m->amp_w / wnorm; // avg=1 over actives
            double xin = xin_contact * w_in_norm;
            double tgt = (fabs(xin) > 1e-9) ? 1.0 : 0.0;
            double env = m->env;
            if (tgt > env) env = tgt + (env - tgt) * att; else env = tgt + (env - tgt) * dec;
            m->env = env;

            double y0 = m->a1*y1 + m->a2*y2 + xin;
            y0_buf[k-1] = y0;
        }

        // ---- pass 2: velocity rotation coupling (energy-preserving), freq-aware to avoid LF blowups
        for (int k=1; k<=N; ++k) v_buf[k-1] = y0_buf[k-1] - y1_prev[k-1];

        if (theta_eased > 1e-9 && N>1){
            int start = (i & 1)? 2 : 1; // alternate pairing each sample
            for (int pair=start; pair < N; pair += 2){
                int k1 = pair, k2 = pair+1;
                if (!active_mask[k1-1] && !active_mask[k2-1]) continue;

                double frel1 = x->m[k1-1].freq_hz / (x->base_hz>0?x->base_hz:1.0);
                double frel2 = x->m[k2-1].freq_hz / (x->base_hz>0?x->base_hz:1.0);
                if (frel1 < 0.1) frel1 = 0.1; if (frel2 < 0.1) frel2 = 0.1;
                double fweight = pow(fmin(frel1, frel2), 0.35); // low freqs -> smaller angle
                double th = theta_eased * fweight;

                double c = cos(th), s = sin(th);
                double a = v_buf[k1-1];
                double b = v_buf[k2-1];
                double A = c*a + s*b;
                double B = -s*a + c*b;
                v_buf[k1-1] = A;
                v_buf[k2-1] = B;
            }
            for (int k=1; k<=N; ++k){ y0_buf[k-1] = y1_prev[k-1] + v_buf[k-1]; }
        }

        // ---- pass 3: state update and output mix (gain is stable 0..1)
        double accL = 0.0, accR = 0.0;
        for (int k=1; k<=N; ++k){
            t_mode *m = &x->m[k-1];
            if (!active_mask[k-1]) continue;
            double y1 = y0_buf[k-1];
            double y2 = y1_prev[k-1];

            double yamp = (double)m->gain * (double)x->mix_scale * m->env * y1;

            accL += yamp * m->gl;
            accR += yamp * m->gr;

            m->y2 = y2;
            m->y1 = y1;
        }

        // apply linear headroom normalization to avoid any clip/drive
        double peak = fmax(fabs(accL), fabs(accR));
        double scl = 1.0;
        const double HEADROOM = 0.98; // keep under float [-1,1]
        if (peak > HEADROOM) scl = HEADROOM / peak;

        outL[i] = (t_sample)(accL * scl);
        outR[i] = (t_sample)(accR * scl);
    }

    return (t_int *)(w+7);
}

static void juicy_bank_tilde_dsp(t_juicy_bank_tilde *x, t_signal **sp){
    x->sr = sp[0]->s_sr;
    dsp_add(juicy_bank_tilde_perform, 6, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[3]->s_vec, sp[0]->s_n);
}

// ---------- setters (body) ----------
static void set_damping   (t_juicy_bank_tilde *x, t_floatarg f){ x->damping=f; }
static void set_brightness(t_juicy_bank_tilde *x, t_floatarg f){ x->brightness=f; }
static void set_position  (t_juicy_bank_tilde *x, t_floatarg f){ x->position=f; }
static void set_dispersion(t_juicy_bank_tilde *x, t_floatarg f){ x->dispersion=f; }
static void set_coupling  (t_juicy_bank_tilde *x, t_floatarg f){ x->coupling=f; }
static void set_density   (t_juicy_bank_tilde *x, t_floatarg f){ x->density=f; }
static void set_anisotropy(t_juicy_bank_tilde *x, t_floatarg f){ x->anisotropy=f; }
static void set_contact   (t_juicy_bank_tilde *x, t_floatarg f){ x->contact=f; }

// ---------- per-mode (via edit cursor) ----------
static void bank_emit_all(t_juicy_bank_tilde *x){ bank_emit_body_state(x); bank_emit_res_selected(x); }
static void set_idx    (t_juicy_bank_tilde *x, t_floatarg f){ int i=(int)floor(f+0.5f); if(i<1)i=1; if(i>x->N)i=x->N; x->edit_idx=i; bank_update_coeffs_one(x,i); bank_emit_res_selected(x); }
static void set_ratio  (t_juicy_bank_tilde *x, t_floatarg f){ int k=x->edit_idx; if(k<1||k>x->N)return; if(f<=0)f=0.0001f; x->m[k-1].ratio=(float)f; x->m[k-1].dirty_coeffs=1; bank_update_coeffs_one(x,k); bank_emit_res_selected(x); }
static void set_gain   (t_juicy_bank_tilde *x, t_floatarg f){ int k=x->edit_idx; if(k<1||k>x->N)return; x->m[k-1].gain=clampf((float)f,0.f,1.f); bank_emit_all(x); }
static void set_attack (t_juicy_bank_tilde *x, t_floatarg f){ int k=x->edit_idx; if(k<1||k>x->N)return; x->m[k-1].attack_ms=(f<0?0:f); bank_emit_res_selected(x); }
static void set_decay  (t_juicy_bank_tilde *x, t_floatarg f){ int k=x->edit_idx; if(k<1||k>x->N)return; x->m[k-1].decay_ms=(f<0?0:f); x->m[k-1].dirty_coeffs=1; bank_update_coeffs_one(x,k); bank_emit_res_selected(x); }
static void set_pan    (t_juicy_bank_tilde *x, t_floatarg f){ int k=x->edit_idx; if(k<1||k>x->N)return; x->m[k-1].pan=clampf((float)f,-1.f,1.f); mode_update_pan(&x->m[k-1]); bank_emit_res_selected(x); }
static void set_keytr  (t_juicy_bank_tilde *x, t_floatarg f){ int k=x->edit_idx; if(k<1||k>x->N)return; x->m[k-1].keytrack=(f!=0)?1.f:0.f; x->m[k-1].dirty_coeffs=1; bank_update_coeffs_one(x,k); bank_emit_res_selected(x); }

// ---------- messages ----------
static void msg_base(t_juicy_bank_tilde *x, t_floatarg f){ x->base_hz=(f>0?f:440.f); for(int k=1;k<=x->N;++k){ x->m[k-1].dirty_coeffs=1; bank_update_coeffs_one(x,k);} bank_emit_all(x); }
static void msg_N(t_juicy_bank_tilde *x, t_floatarg f){
    int oldN=x->N; int n=(int)f; if(n<1)n=1; if(n>MAXN)n=MAXN; x->N=n;
    if(n>oldN){
        for(int k=oldN+1;k<=n;++k){
            t_mode *m=&x->m[k-1];
            m->ratio=1.f; m->gain=0.f; m->attack_ms=0.f; m->decay_ms=600.f; m->pan=0.f; m->keytrack=1.f;
            m->disp_rand = rand_uni_pm1((unsigned int)(1234567 + k*271 + oldN*17));
            m->a1=m->a2=m->y1=m->y2=m->env=0.0; m->gl=1.0; m->gr=0.0; m->freq_hz=0.0; m->rcur=0.0;
            m->amp_w=1.f; m->r_w=1.f; m->pos_w=1.f; m->aniso_w=1.f; m->bright_w=1.f;
            m->cstate=0.0;
            m->active=0; m->dirty_coeffs=1;
            bank_update_coeffs_one(x,k);
        }
    }
    if(x->edit_idx>x->N)x->edit_idx=x->N;
    bank_emit_all(x);
}
static void msg_anisoP(t_juicy_bank_tilde *x, t_floatarg f){ int p=(int)f; if(p<1)p=1; x->aniso_P=p; for(int k=1;k<=x->N;++k){ x->m[k-1].dirty_coeffs=1; bank_update_coeffs_one(x,k);} bank_emit_all(x); }
static void msg_contact_soft(t_juicy_bank_tilde *x, t_floatarg f){ x->contact_soft=clampf(f,0.5f,10.f); bank_emit_body_state(x); }
static void msg_reset(t_juicy_bank_tilde *x){ for(int k=1;k<=x->N;++k){ x->m[k-1].y1=0; x->m[k-1].y2=0; x->m[k-1].env=0; x->m[k-1].cstate=0; } }
static void msg_bang(t_juicy_bank_tilde *x){ bank_emit_all(x); }
static void msg_debug(t_juicy_bank_tilde *x, t_floatarg f){
    x->debug=(f!=0); if(!x->debug) return;
    post("[juicy_bank~] base=%.2f N=%d damp=%.3f bright=%.3f pos=%.3f disp=%.3f dens=%.3f aniso=%.3f P=%d contact=%.3f soft=%.3f",
         x->base_hz, x->N, x->damping, x->brightness, x->position, x->dispersion, x->density, x->anisotropy, x->aniso_P, x->contact, x->contact_soft);
    for(int k=1;k<=x->N;++k){
        t_mode *m=&x->m[k-1];
        post("  id=%d act=%d f=%.2fHz r=%.6f ratio=%.3f key=%d gain=%.3f att=%.1fms dec=%.1fms pan=%.2f posW=%.3f aniso=%.3f bright=%.3f",
             k, m->active, m->freq_hz, m->rcur, m->ratio, (int)(m->keytrack!=0), m->gain, m->attack_ms, m->decay_ms, m->pan,
             m->pos_w, m->aniso_w, m->bright_w);
    }
}
static void msg_bug_catch(t_juicy_bank_tilde *x){
    post("=== juicy_bank~ bug_catch === fs=%.1f base=%.2f N=%d damp=%.3f bright=%.3f pos=%.3f disp=%.3f dens=%.3f aniso=%.3f P=%d contact=%.3f soft=%.3f couple_df=%.1f",
         (double)x->sr, x->base_hz, x->N, x->damping, x->brightness, x->position, x->dispersion, x->density, x->anisotropy, x->aniso_P, x->contact, x->contact_soft, x->couple_df);
    post("n_active=%d mix_scale=%.3f edit_idx=%d", x->n_active, x->mix_scale, x->edit_idx);
    for(int k=1;k<=x->N;++k){
        t_mode *m=&x->m[k-1];
        post("id=%d act=%d f=%.2fHz r=%.6f ratio=%.3f key=%d gain=%.3f att=%.1fms dec=%.1fms pan=%.2f posW=%.3f aniso=%.3f bright=%.3f",
             k, m->active, m->freq_hz, m->rcur, m->ratio, (int)(m->keytrack!=0), m->gain, m->attack_ms, m->decay_ms, m->pan,
             m->pos_w, m->aniso_w, m->bright_w);
    }
    bank_emit_all(x);
}

// ---- Presets (for quick testing) ----
static void preset_apply(t_juicy_bank_tilde *x, t_symbol *name){
    const char *n = name? name->s_name : "init";
    // defaults
    x->base_hz=220; x->N = (x->N>12?12:x->N);
    x->damping=0.15f; x->brightness=0.0f; x->position=0.37f; x->dispersion=0.02f;
    x->coupling=0.20f; x->density=0.0f; x->anisotropy=0.0f; x->contact=0.10f;
    x->aniso_P=2; x->contact_soft=2.0f;

    for (int k=1;k<=x->N;++k){
        t_mode *m=&x->m[k-1];
        m->ratio= (double)k;
        m->gain = (k==1)?1.0f: 0.6f/powf((float)k, 0.75f);
        m->decay_ms = 400.f + 60.f * k;
        m->attack_ms= 0.f;
        m->pan=0.f; m->keytrack=1.f;
    }

    if (!strcmp(n,"marimba")){
        x->brightness=-0.15f; x->position=0.22f; x->dispersion=0.04f; x->coupling=0.25f;
        float barish[12]={1.00f,3.99f,10.94f,21.98f,36.90f,55.66f,78.33f,105.0f,135.6f,170.3f,209.0f,251.7f};
        for(int k=1;k<=x->N && k<=12;++k){ t_mode *m=&x->m[k-1]; m->ratio=barish[k-1]; m->gain=(k==1)?1.0f:0.55f/powf(k,0.6f); m->decay_ms=600+40*k; }
    } else if (!strcmp(n,"glass")){
        x->brightness=+0.35f; x->dispersion=0.08f; x->position=0.31f; x->coupling=0.22f; x->contact=0.0f;
        for(int k=1;k<=x->N;++k){ t_mode *m=&x->m[k-1]; m->gain = (k==1)?0.9f:0.8f/powf(k,0.5f); m->decay_ms = 1200+30*k; }
    } else if (!strcmp(n,"steel")){
        x->brightness=+0.20f; x->dispersion=0.06f; x->position=0.27f; x->coupling=0.28f; x->contact=0.15f;
        for(int k=1;k<=x->N;++k){ t_mode *m=&x->m[k-1]; m->gain = (k==1)?1.0f:0.7f/powf(k,0.55f); m->decay_ms = 1500+20*k; }
    } else if (!strcmp(n,"wood")){
        x->brightness=-0.30f; x->dispersion=0.01f; x->position=0.40f; x->coupling=0.18f; x->contact=0.0f;
        for(int k=1;k<=x->N;++k){ t_mode *m=&x->m[k-1]; m->gain = (k==1)?1.0f:0.5f/powf(k,0.8f); m->decay_ms = 700+50*k; }
    } else if (!strcmp(n,"bell")){
        x->brightness=+0.10f; x->dispersion=0.09f; x->position=0.36f; x->coupling=0.24f; x->contact=0.05f;
        float bellish[12]={1.00f,2.00f,2.40f,3.00f,3.46f,4.00f,5.19f,5.40f,6.80f,8.00f,9.40f,10.80f};
        for(int k=1;k<=x->N && k<=12;++k){ t_mode *m=&x->m[k-1]; m->ratio=bellish[k-1]; m->gain=(k==1)?1.0f:0.6f/powf(k,0.7f); m->decay_ms=1400+40*k; }
    }

    for(int k=1;k<=x->N;++k){ x->m[k-1].dirty_coeffs=1; x->m[k-1].cstate=0; bank_update_coeffs_one(x,k); }
    x->edit_idx=1;
    bank_emit_body_state(x);
    bank_emit_res_selected(x);
}

static void preset_dump(t_juicy_bank_tilde *x, t_symbol *name){
    const char *n = name? name->s_name : "current";
    // print to console
    post("[juicy_bank~] preset_state %s base_hz %.2f N %d damp %.3f bright %.3f pos %.3f disp %.3f coup %.3f dens %.3f aniso %.3f P %d contact %.3f soft %.2f",
         n, x->base_hz, x->N, x->damping, x->brightness, x->position, x->dispersion, x->coupling, x->density, x->anisotropy, x->aniso_P, x->contact, x->contact_soft);
    for (int k=1;k<=x->N;++k){
        t_mode *m=&x->m[k-1];
        post("  id=%d ratio %.5f gain %.5f att_ms %.1f dec_ms %.1f pan %.3f key %d",
             k, m->ratio, m->gain, m->attack_ms, m->decay_ms, m->pan, (int)(m->keytrack!=0));
    }
    // also emit through body outlet as a structured list
    t_atom av[2]; int ac=0;
    SETSYMBOL(av+ac, name?name:gensym("current")); ac++;
    outlet_anything(x->outBody, gensym("preset_state"), ac, av);
}

// ctor
static void *juicy_bank_tilde_new(t_symbol *s, int argc, t_atom *argv){
    (void)s;
    t_juicy_bank_tilde *x = (t_juicy_bank_tilde*)pd_new(juicy_bank_tilde_class);
    x->N=12; x->base_hz=440;
    x->damping=0; x->brightness=0; x->position=0.5f; x->dispersion=0;
    x->coupling=0; x->density=0; x->anisotropy=0; x->contact=0;
    x->aniso_P=2; x->contact_soft=2; x->couple_df=200;
    x->edit_idx=1; x->sr=44100; x->debug=0; x->n_active=0; x->mix_scale=1.f;

    for (int i=0;i<argc;i++){
        if (argv[i].a_type==A_SYMBOL){
            const char *k = atom_getsymbol(argv+i)->s_name;
            if ((!strcmp(k,"@N") || !strcmp(k,"N")) && i+1<argc && argv[i+1].a_type==A_FLOAT){
                int n=(int)atom_getfloat(argv+i+1); if (n<1) n=1; if (n>MAXN) n=MAXN; x->N=n; i++;
            } else if ((!strcmp(k,"@base") || !strcmp(k,"base")) && i+1<argc && argv[i+1].a_type==A_FLOAT){
                float b=atom_getfloat(argv+i+1); if (b>0) x->base_hz=b; i++;
            } else if ((!strcmp(k,"@aniso_P") || !strcmp(k,"aniso_P")) && i+1<argc && argv[i+1].a_type==A_FLOAT){
                int p=(int)atom_getfloat(argv+i+1); if (p<1) p=1; x->aniso_P=p; i++;
            } else if ((!strcmp(k,"@contact_soft") || !strcmp(k,"contact_soft")) && i+1<argc && argv[i+1].a_type==A_FLOAT){
                float cs=atom_getfloat(argv+i+1); if (cs<0.5f) cs=0.5f; if (cs>10.f) cs=10.f; x->contact_soft=cs; i++;
            }
        }
    }

    bank_recalc_body(x);

    for (int k=1; k<=x->N; ++k){
        t_mode *m=&x->m[k-1];
        m->ratio=1.f; m->gain=0.f; m->attack_ms=0.f; m->decay_ms=600.f; m->pan=0.f; m->keytrack=1.f;
        m->disp_rand = rand_uni_pm1((unsigned int)(0xBEEF123 + k*101 + 7));
        m->a1=m->a2=m->y1=m->y2=m->env=0.0; m->gl=1.0; m->gr=0.0; m->freq_hz=0.0; m->rcur=0.0;
        m->amp_w=1.f; m->r_w=1.f; m->pos_w=1.f; m->aniso_w=1.f; m->bright_w=1.f;
        m->cstate=0.0;
        m->active=0; m->dirty_coeffs=1;
        bank_update_coeffs_one(x,k);
    }
    x->m[0].gain=1.f; x->m[0].ratio=1.f; x->m[0].decay_ms=600.f; x->edit_idx=1;

    x->outL  = outlet_new(&x->x_obj, &s_signal);
    x->outR  = outlet_new(&x->x_obj, &s_signal);
    x->outBody = outlet_new(&x->x_obj, &s_anything);
    x->outRes  = outlet_new(&x->x_obj, &s_anything);

    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("signal"), gensym("signal"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("damping"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("brightness"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("position"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("dispersion"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("coupling"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("density"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("anisotropy"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("contact"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("idx"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("ratio"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("gain"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("attack"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("decay"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("pan"));
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("keytrack"));

    bank_emit_body_state(x);
    bank_emit_res_selected(x);
    return (void*)x;
}

void juicy_bank_tilde_setup(void){
    juicy_bank_tilde_class = class_new(gensym("juicy_bank~"),
        (t_newmethod)juicy_bank_tilde_new, 0,
        sizeof(t_juicy_bank_tilde), CLASS_DEFAULT, A_GIMME, 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dsp, gensym("dsp"), A_CANT, 0);
    CLASS_MAINSIGNALIN(juicy_bank_tilde_class, t_juicy_bank_tilde, f_dummy);

    class_addmethod(juicy_bank_tilde_class, (t_method)set_damping,    gensym("damping"),    A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)set_brightness, gensym("brightness"), A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)set_position,   gensym("position"),   A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)set_dispersion, gensym("dispersion"), A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)set_coupling,   gensym("coupling"),   A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)set_density,    gensym("density"),    A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)set_anisotropy, gensym("anisotropy"), A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)set_contact,    gensym("contact"),    A_FLOAT, 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)set_idx,    gensym("idx"),    A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)set_ratio,  gensym("ratio"),  A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)set_gain,   gensym("gain"),   A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)set_attack, gensym("attack"), A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)set_decay,  gensym("decay"),  A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)set_pan,    gensym("pan"),    A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)set_keytr,  gensym("keytrack"),A_FLOAT,0);

    class_addmethod(juicy_bank_tilde_class, (t_method)msg_base,         gensym("base"),         A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)msg_N,            gensym("N"),            A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)msg_anisoP,       gensym("aniso_P"),      A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)msg_contact_soft, gensym("contact_soft"), A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)msg_reset,        gensym("reset"),        0);
    class_addmethod(juicy_bank_tilde_class, (t_method)msg_bang,         gensym("bang"),         0);
    class_addmethod(juicy_bank_tilde_class, (t_method)msg_debug,        gensym("debug"),        A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)msg_bug_catch,    gensym("bug_catch"),    0);

    class_addmethod(juicy_bank_tilde_class, (t_method)preset_apply, gensym("preset"), A_SYMBOL, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)preset_dump,  gensym("preset_dump"), A_DEFSYM, 0);
}
