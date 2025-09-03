// juicy_bank_tilde.c — Juicy's modal **bank** (N resonators in one object)
// PATCH: 2025-09-03 (pass 2)
// Fixes based on Juicy's 5-point feedback:
// 1) Damping mapping re-shaped: 0..1 is the *intended* range. Smooth, not twitchy.
// 2) Dispersion stays per-mode random, unique per resonator; knob sets max spread (±0.9 * |knob|).
// 3) Coupling reworked to a *diffusive, stable* form with frequency-falloff and damping scaling.
// 4) Contact loudness: revised makeup gain (calibrated for typical amplitudes) to avoid the dip near 0+.
// 5) Even indices silent at pos=0.5 explained by physics; add a higher floor (0.02) so they still whisper.
//
// Ranges you should use in the patcher (all clamped here):
// damping [0..1], brightness [-1..+1], position [0..1], dispersion [-1..+1], coupling [0..1], density [-1..+1], anisotropy [-1..+1], contact [0..1].

#include "m_pd.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAXN 128

static t_class *juicy_bank_tilde_class;

static inline float clampf(float x, float lo, float hi){ if(x<lo) return lo; if(x>hi) return hi; return x; }
static inline double denorm_fix(double v){ return (v<1e-30 && v>-1e-30)?0.0:v; }

typedef struct {
    float ratio, gain, attack_ms, decay_ms, pan, keytrack;
    float disp_rand; // stable per-mode random in [-1,1]
    double a1, a2, y1, y2, env, gl, gr, freq_hz, rcur;
    float amp_w, r_w, pos_w, aniso_w, bright_w;
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
    double th = (p + 1.0) * 0.25 * M_PI;
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

    // 1) New damping mapping: smooth & usable across 0..1
    // Internal scale: d_s = (d)^2 compresses near 0, keeps 1 = full.
    // r_mul_base in [0.8 .. 1.0] then shaped by a *mild* HF exponent.
    float d_s = x->damping * x->damping;
    x->r_mul_base = 1.f - 0.20f * d_s;     // 1 → 0.8 at full damping (mild base)
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
    // 2) Dispersion: unique per mode via disp_rand, knob sets *max spread*
    float disp_amt = 0.9f * fabsf(x->dispersion); // 0..0.9
    double ratio_eff = (double)m->ratio;
    if (m->keytrack!=0.f){
        ratio_eff = (double)m->ratio + (double)disp_amt * (double)m->disp_rand;
        if (ratio_eff <= 0.0001) ratio_eff = 0.0001;
    }
    double fr = (m->keytrack!=0.f) ? base * ratio_eff : (double)m->ratio;
    if (m->keytrack==0.f){
        fr = (double)m->ratio * (1.0 + (double)disp_amt * (double)m->disp_rand);
        if (fr < 1.0) fr = 1.0;
    }

    // Density: cluster/spread around base (or ratio=1)
    float dens = x->density;
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

    // 1) Damping: map decay_ms to r, then mild HF shaping by frel
    double d_s = (m->decay_ms<=0?0.0:m->decay_ms*0.001);
    double r = (d_s<=0.0)? 0.0 : exp(-1.0/(d_s*fs));
    double frel = fr / base; if (frel < 0.1) frel = 0.1;
    double expo = 1.0 + 0.5 * sqrt(frel);         // mild HF emphasis on damping
    double r_w = pow((double)x->r_mul_base, expo); // x->r_mul_base in [0.8..1.0]
    r *= r_w;
    if (r>0.999999) r = 0.999999;
    if (r<0.0)      r = 0.0;
    m->rcur = r;

    double w = 2.0 * M_PI * fr / fs;
    m->a1 = 2.0 * r * cos(w);
    m->a2 = - (r * r);

    // Position: true node behavior, but with a small protective floor (0.02)
    int    idx1 = k;
    double posw = fabs(sin(M_PI * (double)idx1 * (double)clampf(x->position,0.f,1.f)));
    if (posw < 0.02) posw = 0.02; // keep even modes faintly audible at pos=0.5

    // Anisotropy: attenuation-only for opposite group
    int P = (x->aniso_P<1?1:x->aniso_P);
    double cyc = cos(2.0*M_PI * (double)(idx1-1) / (double)P);
    double a = x->aniso_gamma;
    double aniso = 1.0;
    if (a > 0){
        if (cyc < 0) aniso = 1.0 - (double)fabs(a) * 0.8;
    } else if (a < 0){
        if (cyc > 0) aniso = 1.0 - (double)fabs(a) * 0.8;
    }

    // Brightness (tempered) — normalized later
    double bright = pow(fr/base, (double)x->bright_slope);
    if (bright < 0.05) bright = 0.05;
    if (bright > 20.0) bright = 20.0;

    m->pos_w    = (float)posw;
    m->aniso_w  = (float)aniso;
    m->bright_w = (float)bright;
    m->dirty_coeffs = 0;

    mode_update_pan(m);
}

static inline double contact_makeup(double amt, double soft){
    if (amt<=0.0) return 1.0;
    // Calibrate around a moderate internal amplitude to avoid a dip just above 0
    double k = amt / (soft>1e-6?soft:1.0);
    double a_cal = 0.75; // calibration amplitude
    double g = tanh(k*a_cal) / (k*a_cal);
    if (g < 1e-3) g = 1e-3;
    double mk = 1.0 / g;
    if (mk > 8.0) mk = 8.0;
    return mk;
}
static inline double contact_shaper(double x, double amt, double soft){
    if (amt<=0.0) return x;
    double k = amt / (soft>1e-6?soft:1.0);
    return tanh(k*x);
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

    // Brightness normalization + fair gain
    double bright_norm = (bright_sum>0.0)? (bright_sum / (double)(nact>0?nact:1)) : 1.0;
    if (bright_norm <= 0.0) bright_norm = 1.0;
    double ampW_sum = 0.0;
    for (int k=1; k<=N; ++k){
        t_mode *m = &x->m[k-1];
        double ampw = (double)m->pos_w * (double)m->aniso_w * ((double)m->bright_w / bright_norm);
        if (ampw < 0.0) ampw = 0.0;
        if (ampw > 32.0) ampw = 32.0;
        m->amp_w = (float)ampw;
        if (m->active) ampW_sum += ampw;
    }
    double amp_norm = (nact>0)? ((double)nact / (ampW_sum + 1e-12)) : 1.0;

    // Diffusive coupling: pull each mode slightly toward neighbors (stable)
    double coup = (double)x->coupling;
    double dfc  = (double)x->couple_df;
    double kL[MAXN], kR[MAXN];
    for (int k=1; k<=N; ++k){
        double gL=0.0, gR=0.0;
        if (coup>1e-6 && N>1){
            if (k>1){
                double df = fabs(x->m[k-2].freq_hz - x->m[k-1].freq_hz);
                gL = exp( - df / (dfc>1e-9?dfc:1.0) );
            }
            if (k<N){
                double df = fabs(x->m[k].freq_hz - x->m[k-1].freq_hz);
                gR = exp( - df / (dfc>1e-9?dfc:1.0) );
            }
        }
        kL[k-1]=(float)gL; kR[k-1]=(float)gR;
    }

    for (int i=0;i<n;++i){ outL[i]=0; outR[i]=0; }

    const double c_amt  = x->contact;
    const double c_soft = x->contact_soft;
    const double c_mkup = contact_makeup(c_amt, c_soft);

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
            double att = (m->attack_ms<=0? 0.0 : exp(-1.0/( (double)m->attack_ms*0.001 * fs )));
            double dec = (m->decay_ms <=0? 0.0 : exp(-1.0/( (double)m->decay_ms *0.001 * fs )));
            double gl=m->gl, gr=m->gr;

            double xin = xin_mono * (double)m->pos_w;
            double tgt = (fabs(xin) > 1e-9) ? 1.0 : 0.0;
            if (tgt > env) env = tgt + (env - tgt) * att; else env = tgt + (env - tgt) * dec;

            // diffusive coupling: proportional to (neighbor - self), scaled by damping
            double inj = 0.0;
            if (coup>1e-6 && N>1){
                double damp_scale = (1.0 - m->rcur); // less coupling when very resonant (stability)
                double g = coup * 0.01 * damp_scale; // base coefficient
                if (k>1) inj += g * kL[k-1] * (y1_old[k-2] - y1);
                if (k<N) inj += g * kR[k-1] * (y1_old[k]   - y1);
                if (inj > 0.1) inj = 0.1; else if (inj < -0.1) inj = -0.1;
            }

            double y0 = a1*y1 + a2*y2 + xin + inj;
            y2 = y1; y1 = y0;

            double ynl = (c_amt>1e-6)? contact_shaper(y0, c_amt, c_soft) : y0;
            double amp = (double)m->gain * (double)x->mix_scale * (double)amp_norm * (double)m->amp_w;
            double yamp= amp * env * ynl * c_mkup;

            outL[i] = (t_sample)denorm_fix( (double)outL[i] + yamp * gl );
            outR[i] = (t_sample)denorm_fix( (double)outR[i] + yamp * gr );

            m->y1=y1; m->y2=y2; m->env=env;
        }
    }

    return (t_int *)(w+7);
}

static void juicy_bank_tilde_dsp(t_juicy_bank_tilde *x, t_signal **sp){
    x->sr = sp[0]->s_sr;
    dsp_add(juicy_bank_tilde_perform, 6, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[3]->s_vec, sp[0]->s_n);
}

// setters (body)
static void set_damping   (t_juicy_bank_tilde *x, t_floatarg f){ x->damping=f; }
static void set_brightness(t_juicy_bank_tilde *x, t_floatarg f){ x->brightness=f; }
static void set_position  (t_juicy_bank_tilde *x, t_floatarg f){ x->position=f; }
static void set_dispersion(t_juicy_bank_tilde *x, t_floatarg f){ x->dispersion=f; }
static void set_coupling  (t_juicy_bank_tilde *x, t_floatarg f){ x->coupling=f; }
static void set_density   (t_juicy_bank_tilde *x, t_floatarg f){ x->density=f; }
static void set_anisotropy(t_juicy_bank_tilde *x, t_floatarg f){ x->anisotropy=f; }
static void set_contact   (t_juicy_bank_tilde *x, t_floatarg f){ x->contact=f; }

// per-mode (via edit cursor)
static void bank_emit_all(t_juicy_bank_tilde *x){ t_atom a; (void)a; bank_emit_body_state(x); bank_emit_res_selected(x); }
static void set_idx    (t_juicy_bank_tilde *x, t_floatarg f){ int i=(int)f; if(i<1)i=1; if(i>x->N)i=x->N; x->edit_idx=i; bank_update_coeffs_one(x,i); bank_emit_res_selected(x); }
static void set_ratio  (t_juicy_bank_tilde *x, t_floatarg f){ int k=x->edit_idx; if(k<1||k>x->N)return; if(f<=0)f=0.0001f; x->m[k-1].ratio=(float)f; x->m[k-1].dirty_coeffs=1; bank_update_coeffs_one(x,k); bank_emit_res_selected(x); }
static void set_gain   (t_juicy_bank_tilde *x, t_floatarg f){ int k=x->edit_idx; if(k<1||k>x->N)return; x->m[k-1].gain=clampf((float)f,0.f,1.f); bank_emit_all(x); }
static void set_attack (t_juicy_bank_tilde *x, t_floatarg f){ int k=x->edit_idx; if(k<1||k>x->N)return; x->m[k-1].attack_ms=(f<0?0:f); bank_emit_res_selected(x); }
static void set_decay  (t_juicy_bank_tilde *x, t_floatarg f){ int k=x->edit_idx; if(k<1||k>x->N)return; x->m[k-1].decay_ms=(f<0?0:f); x->m[k-1].dirty_coeffs=1; bank_update_coeffs_one(x,k); bank_emit_res_selected(x); }
static void set_pan    (t_juicy_bank_tilde *x, t_floatarg f){ int k=x->edit_idx; if(k<1||k>x->N)return; x->m[k-1].pan=clampf((float)f,-1.f,1.f); mode_update_pan(&x->m[k-1]); bank_emit_res_selected(x); }
static void set_keytr  (t_juicy_bank_tilde *x, t_floatarg f){ int k=x->edit_idx; if(k<1||k>x->N)return; x->m[k-1].keytrack=(f!=0)?1.f:0.f; x->m[k-1].dirty_coeffs=1; bank_update_coeffs_one(x,k); bank_emit_res_selected(x); }

// messages
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
            m->active=0; m->dirty_coeffs=1;
            bank_update_coeffs_one(x,k);
        }
    }
    if(x->edit_idx>x->N)x->edit_idx=x->N;
    bank_emit_all(x);
}
static void msg_anisoP(t_juicy_bank_tilde *x, t_floatarg f){ int p=(int)f; if(p<1)p=1; x->aniso_P=p; for(int k=1;k<=x->N;++k){ x->m[k-1].dirty_coeffs=1; bank_update_coeffs_one(x,k);} bank_emit_all(x); }
static void msg_contact_soft(t_juicy_bank_tilde *x, t_floatarg f){ x->contact_soft=clampf(f,0.5f,10.f); bank_emit_body_state(x); }
static void msg_reset(t_juicy_bank_tilde *x){ for(int k=1;k<=x->N;++k){ x->m[k-1].y1=0; x->m[k-1].y2=0; x->m[k-1].env=0; } }
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
}
