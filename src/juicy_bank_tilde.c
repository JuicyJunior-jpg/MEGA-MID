// juicy_bank_tilde.c — Juicy's modal **bank** (N resonators in one object)
// PATCH: 2025-09-03 (pass 5)
// Changes this pass:
//  • Coupling: switched to **velocity coupling** (waveguide-ish), with per‑mode LP on the injection.
//  • Contact: changed from crossfade to **additive injection** of the nonlinear term with level/low‑freq comp.
//  • Presets: new message `preset <symbol>` to load built-ins; `preset_dump` to print/emit current state.
//  • HD: compile-time flag JUICY_HD for slightly tighter numerics (keeps tone; just lower noise).
//
// Expected knob ranges (clamped here):
// damping [0..1], brightness [-1..+1], position [0..1], dispersion [-1..+1], coupling [0..1], density [-1..+1], anisotropy [-1..+1], contact [0..1].
// Indices are 1-based externally.

#include "m_pd.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef JUICY_HD
#define JUICY_HD 0   // set to 1 to enable slightly stronger internal smoothing (no tone change intended)
#endif

#define MAXN 128

static t_class *juicy_bank_tilde_class;

static inline float clampf(float x, float lo, float hi){ return (x<lo)?lo:((x>hi)?hi:x); }
static inline double denorm_fix(double v){ return (v<1e-30 && v>-1e-30)?0.0:v; }

typedef struct {
    float ratio, gain, attack_ms, decay_ms, pan, keytrack;
    float disp_rand; // stable per-mode random in [-1,1]

    // biquad
    double a1, a2, y1, y2, env, freq_hz, rcur;
    // panner
    double gl, gr;

    // weights
    float amp_w, r_w, pos_w, aniso_w, bright_w;

    // coupling state (per-mode LP for injection)
    double cstate;

    unsigned char active, dirty_coeffs;
} t_mode;

typedef struct _juicy_bank_tilde {
    t_object  x_obj;
    t_float   f_dummy;

    int     N;
    float   base_hz;
    float   damping, brightness, position, dispersion, coupling, density, anisotropy, contact;
    int     aniso_P;
    float   contact_soft, couple_df;

    // derived
    float   r_mul_base, bright_slope, aniso_gamma;
    float   mix_scale;
    int     n_active;

    int     edit_idx;
    t_mode  m[MAXN];

    float   sr;
    int     debug;

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

    // Damping map 0..1 -> pole base multiplier (mild, freq-shaped later)
    float d_s = x->damping * x->damping;
    x->r_mul_base = 1.f - 0.20f * d_s;     // 1 → 0.8 at full damping (global)
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
    SETSYMBOL(av+i, gensym("amp_w")); i++; SETFLOAT(av+i, m->amp_w); i++;
    SETSYMBOL(av+i, gensym("r_w")); i++; SETFLOAT(av+i, m->r_w); i++;
    SETSYMBOL(av+i, gensym("pos_w")); i++; SETFLOAT(av+i, m->pos_w); i++;
    SETSYMBOL(av+i, gensym("aniso_w")); i++; SETFLOAT(av+i, m->aniso_w); i++;
    SETSYMBOL(av+i, gensym("bright_w")); i++; SETFLOAT(av+i, m->bright_w); i++;
    outlet_anything(x->outRes, gensym("res_state"), i, av);
}

static void bank_update_coeffs_one(t_juicy_bank_tilde *x, int k){
    if (k < 1 || k > x->N) return;
    t_mode *m = &x->m[k-1];

    double base = (x->base_hz>0?x->base_hz:1.0);
    // Dispersion: unique per mode via disp_rand, knob sets *max spread* (±0.9*|knob|)
    float disp_amt = 0.9f * fabsf(x->dispersion); // 0..0.9 absolute offset
    double ratio_eff = (double)m->ratio;
    if (m->keytrack!=0.f){
        // Do NOT disperse fundamental (k==1)
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

    // Position: true node behavior, + floor (0.10)
    int    idx1 = k;
    double posw = fabs(sin(M_PI * (double)idx1 * (double)clampf(x->position,0.f,1.f)));
    if (posw < 0.10) posw = 0.10;

    // Anisotropy: attenuation-only for opposite group (period P)
    int P = (x->aniso_P<1?1:x->aniso_P);
    double cyc = cos(2.0*M_PI * (double)(idx1-1) / (double)P);
    double a = x->aniso_gamma;
    double aniso = 1.0;
    if (a > 0){
        if (cyc < 0) aniso = 1.0 - (double)fabs(a) * 0.8;
    } else if (a < 0){
        if (cyc > 0) aniso = 1.0 - (double)fabs(a) * 0.8;
    }

    // Brightness (tempered) — normalized later per-block
    double bright = pow(fr/base, (double)x->bright_slope);
    if (bright < 0.05) bright = 0.05;
    if (bright > 20.0) bright = 20.0;

    m->pos_w    = (float)posw;
    m->aniso_w  = (float)aniso;
    m->bright_w = (float)bright;
    m->dirty_coeffs = 0;

    mode_update_pan(m);
}

// nonlinear contact term (odd-harmonic friendly). Returns added term (not mixed).
static inline double contact_residual(double x, double amt, double soft){
    if (amt<=0.0) return 0.0;
    if (soft<=1e-6) soft = 1e-6;
    double nl = soft * tanh(x/soft);
    // Inject the non-linear **excess** w.r.t. identity, then compensate level slightly:
    double resid = (nl - x);
    double comp  = 1.0 + 0.30*amt; // gentle makeup so it doesn't dip at small amt
    return comp * resid;
}

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
    double bright_sum = 0.0;
    for (int k=1; k<=N; ++k){
        t_mode *m = &x->m[k-1];
        m->active = (m->gain > 0.0005f) || (fabs(m->env) > 1e-6);
        bank_update_coeffs_one(x, k);
        if (m->active){ nact++; bright_sum += (double)m->bright_w; }
    }
    x->n_active = nact;
    x->mix_scale = (nact>0)? (1.0f / sqrtf((float)nact)) : 1.0f;

    // Brightness normalization + fair gain + position compensation
    double bright_norm = (bright_sum>0.0)? (bright_sum / (double)(nact>0?nact:1)) : 1.0;
    if (bright_norm <= 0.0) bright_norm = 1.0;
    double ampW_sum = 0.0;
    for (int k=1; k<=N; ++k){
        t_mode *m = &x->m[k-1];
        // position compensation: reduce loudness gap for near-nodal modes
        double p = (double)m->pos_w;
        double pos_comp = pow((p<1e-6?1e-6:p)/0.5, -0.5); // ~sqrt(0.5/p)
        if (pos_comp > 2.0) pos_comp = 2.0;

        double ampw = (double)m->pos_w * (double)m->aniso_w * ((double)m->bright_w / bright_norm) * pos_comp;
        if (ampw < 0.0) ampw = 0.0;
        if (ampw > 32.0) ampw = 32.0;
        m->amp_w = (float)ampw;
        if (m->active) ampW_sum += ampw;
    }
    double amp_norm = (nact>0)? ((double)nact / (ampW_sum + 1e-12)) : 1.0;

    for (int i=0;i<n;++i){ outL[i]=0; outR[i]=0; }

    const double c_amt  = x->contact;
    const double c_soft = x->contact_soft;

    for (int i=0;i<n;++i){
        double y1_old[MAXN];
        for (int k=1; k<=N; ++k){ y1_old[k-1]=x->m[k-1].y1; }

        double xin_mono = 0.5*((double)inL[i] + (double)inR[i]);

        for (int k=1; k<=N; ++k){
            t_mode *m = &x->m[k-1];
            if (!m->active) continue;

            double a1=m->a1, a2=m->a2, y1=m->y1, y2=m->y2;
            double env=m->env;
            double fs = (double)x->sr;
            // AR env
#if JUICY_HD
            double att = (m->attack_ms<=0? 0.0 : exp(-1.0/( (double)m->attack_ms*0.001 * fs )));
            double dec = (m->decay_ms <=0? 0.0 : exp(-1.0/( (double)m->decay_ms *0.001 * fs )));
#else
            double att = (m->attack_ms<=0? 0.0 : exp(-1.0/( (double)m->attack_ms*0.001 * fs )));
            double dec = (m->decay_ms <=0? 0.0 : exp(-1.0/( (double)m->decay_ms *0.001 * fs )));
#endif
            double gl=m->gl, gr=m->gr;

            double xin = xin_mono * (double)m->pos_w;
            double tgt = (fabs(xin) > 1e-9) ? 1.0 : 0.0;
            if (tgt > env) env = tgt + (env - tgt) * att; else env = tgt + (env - tgt) * dec;

            // core resonator
            double y0 = a1*y1 + a2*y2 + xin;

            // ===== Velocity coupling (neighbors) =====
            double coup = (double)x->coupling;
            if (coup>1e-6 && N>1){
                // self velocity
                double v  = y1 - y2;
                double vinj = 0.0;
                if (k>1){
                    t_mode *ml = &x->m[k-2];
                    double vL = y1_old[k-2] - ml->y2;
                    vinj += (vL - v);
                }
                if (k<N){
                    t_mode *mr = &x->m[k];
                    double vR = y1_old[k] - mr->y2;
                    vinj += (vR - v);
                }
                // frequency-aware falloff (friendly at HF), scaled by damping
                double damp_scale = 0.3 + 0.7 * (1.0 - m->rcur);
                double gbase = 0.10 * (0.5 - 0.5*cos(M_PI*coup)); // eased 0..~0.10
                double g = gbase * damp_scale;
                // light LP on injection per-mode
#if JUICY_HD
                m->cstate = 0.25*vinj + 0.75*m->cstate;
#else
                m->cstate = 0.35*vinj + 0.65*m->cstate;
#endif
                double inj = g * m->cstate;
                if (inj > 0.2) inj = 0.2; else if (inj < -0.2) inj = -0.2;
                y0 += inj;
            }

            // ===== Contact: additive injection of nonlinear residue =====
            if (c_amt>1e-6){
                double resid = contact_residual(y0, c_amt, c_soft);
                // gentle LF compensation so lows don't vanish under contact
                double ratio_f = (m->freq_hz<=0?1.0:(m->freq_hz / (x->base_hz>0?x->base_hz:1.0)));
                if (ratio_f < 1e-6) ratio_f = 1e-6;
                double eq = pow(ratio_f, -0.25 * c_amt);
                if (eq < 0.5) eq = 0.5; if (eq > 2.0) eq = 2.0;
                y0 += eq * resid;
            }

            // state update
            y2 = y1; y1 = y0;

            // pan, weights, env, mix
            double amp = (double)m->gain * (double)x->mix_scale * (double)amp_norm * (double)m->amp_w;
            double yamp= amp * env * y1;

            outL[i] = (t_sample)denorm_fix( (double)outL[i] + yamp * m->gl );
            outR[i] = (t_sample)denorm_fix( (double)outR[i] + yamp * m->gr );

            m->y1=y1; m->y2=y2; m->env=env;
        }
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
        post("  id=%d act=%d f=%.2fHz r=%.6f ratio=%.3f key=%d gain=%.3f att=%.1fms dec=%.1fms pan=%.2f posW=%.3f aniso=%.3f bright=%.3f ampW=%.3f dispRand=%.3f",
             k, m->active, m->freq_hz, m->rcur, m->ratio, (int)(m->keytrack!=0), m->gain, m->attack_ms, m->decay_ms, m->pan,
             m->pos_w, m->aniso_w, m->bright_w, m->amp_w, m->disp_rand);
    }
}
static void msg_bug_catch(t_juicy_bank_tilde *x){
    post("=== juicy_bank~ bug_catch === fs=%.1f base=%.2f N=%d damp=%.3f bright=%.3f pos=%.3f disp=%.3f dens=%.3f aniso=%.3f P=%d contact=%.3f soft=%.3f couple_df=%.1f",
         (double)x->sr, x->base_hz, x->N, x->damping, x->brightness, x->position, x->dispersion, x->density, x->anisotropy, x->aniso_P, x->contact, x->contact_soft, x->couple_df);
    post("n_active=%d mix_scale=%.3f edit_idx=%d", x->n_active, x->mix_scale, x->edit_idx);
    for(int k=1;k<=x->N;++k){
        t_mode *m=&x->m[k-1];
        post("id=%d act=%d f=%.2fHz r=%.6f ratio=%.3f key=%d gain=%.3f att=%.1fms dec=%.1fms pan=%.2f posW=%.3f aniso=%.3f bright=%.3f ampW=%.3f dispRand=%.3f",
             k, m->active, m->freq_hz, m->rcur, m->ratio, (int)(m->keytrack!=0), m->gain, m->attack_ms, m->decay_ms, m->pan,
             m->pos_w, m->aniso_w, m->bright_w, m->amp_w, m->disp_rand);
    }
    bank_emit_all(x);
}

// ---- Presets ----
// Built-ins for quick A/B. You can tweak later.
static void preset_apply(t_juicy_bank_tilde *x, t_symbol *name){
    const char *n = name? name->s_name : "init";
    // defaults
    x->base_hz=220; x->N = (x->N>12?12:x->N);
    x->damping=0.15f; x->brightness=0.0f; x->position=0.37f; x->dispersion=0.02f;
    x->coupling=0.35f; x->density=0.0f; x->anisotropy=0.0f; x->contact=0.10f;
    x->aniso_P=2; x->contact_soft=2.0f;

    // per-mode init
    for (int k=1;k<=x->N;++k){
        t_mode *m=&x->m[k-1];
        m->ratio= (double)k;   // harmonic scaffold
        m->gain = (k==1)?1.0f: 0.6f/powf((float)k, 0.75f);
        m->decay_ms = 400.f + 60.f * k;
        m->attack_ms= 0.f;
        m->pan=0.f; m->keytrack=1.f;
    }

    if (!strcmp(n,"marimba")){
        x->brightness=-0.15f; x->position=0.22f; x->dispersion=0.04f; x->coupling=0.45f;
        float barish[12]={1.00f,3.99f,10.94f,21.98f,36.90f,55.66f,78.33f,105.0f,135.6f,170.3f,209.0f,251.7f};
        for(int k=1;k<=x->N && k<=12;++k){ t_mode *m=&x->m[k-1]; m->ratio=barish[k-1]; m->gain=(k==1)?1.0f:0.55f/powf(k,0.6f); m->decay_ms=600+40*k; }
    } else if (!strcmp(n,"glass")){
        x->brightness=+0.35f; x->dispersion=0.08f; x->position=0.31f; x->coupling=0.30f; x->contact=0.0f;
        for(int k=1;k<=x->N;++k){ t_mode *m=&x->m[k-1]; m->gain = (k==1)?0.9f:0.8f/powf(k,0.5f); m->decay_ms = 1200+30*k; }
    } else if (!strcmp(n,"steel")){
        x->brightness=+0.20f; x->dispersion=0.06f; x->position=0.27f; x->coupling=0.50f; x->contact=0.15f;
        for(int k=1;k<=x->N;++k){ t_mode *m=&x->m[k-1]; m->gain = (k==1)?1.0f:0.7f/powf(k,0.55f); m->decay_ms = 1500+20*k; }
    } else if (!strcmp(n,"wood")){
        x->brightness=-0.30f; x->dispersion=0.01f; x->position=0.40f; x->coupling=0.25f; x->contact=0.0f;
        for(int k=1;k<=x->N;++k){ t_mode *m=&x->m[k-1]; m->gain = (k==1)?1.0f:0.5f/powf(k,0.8f); m->decay_ms = 700+50*k; }
    } else if (!strcmp(n,"bell")){
        x->brightness=+0.10f; x->dispersion=0.09f; x->position=0.36f; x->coupling=0.40f; x->contact=0.05f;
        float bellish[12]={1.00f,2.00f,2.40f,3.00f,3.46f,4.00f,5.19f,5.40f,6.80f,8.00f,9.40f,10.80f};
        for(int k=1;k<=x->N && k<=12;++k){ t_mode *m=&x->m[k-1]; m->ratio=bellish[k-1]; m->gain=(k==1)?1.0f:0.6f/powf(k,0.7f); m->decay_ms=1400+40*k; }
    } else {
        // "init" / unknown: keep defaults above
    }

    // refresh coeffs
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
    t_atom av[4]; int ac=0;
    SETSYMBOL(av+ac, gensym("preset_state")); ac++;
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

    // presets
    class_addmethod(juicy_bank_tilde_class, (t_method)preset_apply, gensym("preset"), A_SYMBOL, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)preset_dump,  gensym("preset_dump"), A_DEFSYM, 0);
}
