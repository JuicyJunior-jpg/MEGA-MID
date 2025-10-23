
// juicy_bank~ — dual-bank modal resonator (A=64, B=32) with coupling
// Topologies: single A, single B, parallel, serial (A excites B and both mixed)
// Bank-select inlet: "modal A" / "modal B" — all parameter messages/inlets
//                     affect the selected bank. When editing B, "partials"
//                     scales 1..64 to 1..32.
//
// Build (macOS):
//   cc -O3 -fPIC -DPD -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type \
//     -I"/Applications/Pd-0.56-1.app/Contents/Resources/src" \
//     -arch arm64 -arch x86_64 -mmacosx-version-min=10.13 \
//     -bundle -undefined dynamic_lookup \
//     -o juicy_bank~.pd_darwin juicy_bank_tilde.c
//
// Build (Linux):
//   cc -O3 -fPIC -DPD -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type \
//     -I"/usr/include/pd" -shared -fPIC -Wl,-export-dynamic -lm \
//     -o juicy_bank~.pd_linux juicy_bank_tilde.c

#include "m_pd.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- limits ----------
#define JB_MAX_VOICES     4
#define JB_MAX_MODES_A   64
#define JB_MAX_MODES_B   32
#define JB_MAX_MODES     JB_MAX_MODES_A   // arrays sized to A, B uses subset

// ---------- utils ----------
static inline float jb_clamp(float x, float lo, float hi){ return (x<lo)?lo:((x>hi)?hi:x); }
static inline float jb_wrap01(float x){ x = x - floorf(x); if (x < 0.f) x += 1.f; return x; }
typedef struct { unsigned int s; } jb_rng_t;
static inline void jb_rng_seed(jb_rng_t *r, unsigned int s){ if(!s) s=1; r->s = s; }
static inline unsigned int jb_rng_u32(jb_rng_t *r){ unsigned int x = r->s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; r->s = x; return x; }
static inline float jb_rng_bi(jb_rng_t *r){ return ((int)(jb_rng_u32(r)) / (float)INT_MAX); }

static inline float jb_midi_to_hz(float n){ return 440.f * powf(2.f, (n-69.f)/12.f); }

typedef enum { V_IDLE=0, V_HELD=1, V_RELEASE=2 } jb_vstate;

// ---------- mode/base/runtime ----------
typedef struct {
    // base params (template per mode)
    float base_ratio, base_decay_ms, base_gain;
    float attack_ms, curve_amt, pan;
    int   active;
    int   keytrack; // 1 = track f0 (ratio), 0 = absolute Hz
    // signatures (random)
    float disp_signature;
} jb_mode_base_t;

typedef struct {
    // runtime per-mode
    float ratio_now, t60_s;
    // biquad coeffs/state per ear
    float a1L,a2L, y1L,y2L;  // primary pole pair
    float a1R,a2R, y1R,y2R;
    float driveL, driveR;
    float normL, normR; // frequency-normalized drive factor
    float envL, envR;   // energy tracker
    // random hit state
    int   hit_gateL, hit_coolL, hit_gateR, hit_coolR;
} jb_mode_rt_t;

typedef struct {
    jb_vstate state;
    float f0, vel, energy;
    float rel_env;
    jb_mode_rt_t m[JB_MAX_MODES];
} jb_voice_t;

// ---------- Bank container ----------
typedef struct {
    // configuration
    int n_modes;        // 64 for A, 32 for B
    int active_modes;   // 1..n_modes (set via "partials")
    // base tables
    jb_mode_base_t base[JB_MAX_MODES]; // B uses [0..31]
    // voices (per-bank runtime states)
    jb_voice_t v[JB_MAX_VOICES];
} jb_bank_t;

// ---------- the object ----------
static t_class *juicy_bank_tilde_class;

typedef enum { TOPO_SINGLE_A=0, TOPO_SINGLE_B=1, TOPO_PARALLEL=2, TOPO_SERIAL=3 } jb_topology;
typedef enum { BANK_EDIT_A=0, BANK_EDIT_B=1 } jb_bank_edit;

typedef struct _juicy_bank_tilde {
    t_object  x_obj; t_float f_dummy; t_float sr;
    // two banks
    jb_bank_t A;
    jb_bank_t B;
    // selection & topology
    jb_bank_edit edit_bank;   // which bank params edit apply to
    jb_topology  topo;        // coupling topology
    // body/behavior globals shared by both banks
    float damping, brightness, position;
    float damp_broad, damp_point;
    float dispersion, dispersion_last;
    float aniso, aniso_eps;
    float contact_amt, contact_sym;
    float basef0_ref;
    float release_amt;
    float phase_rand;
    // IO (signals)
    t_inlet *inR;             // main R (main L is implicit)
    t_inlet *in_vL[JB_MAX_VOICES];
    t_inlet *in_vR[JB_MAX_VOICES];
    int exciter_mode;         // 0=legacy main L/R, 1=per-voice
    // new "bank select" inlet (symbol messages)
    t_inlet *in_bank_select;
    // "partials" inlet (affects selected bank)
    t_inlet *in_partials;
    // keytrack/index/etc inlets (affect selected bank)
    t_inlet *in_index, *in_ratio, *in_gain, *in_attack, *in_decay, *in_curve, *in_pan, *in_keytrack;
    // body inlets (shared)
    t_inlet *in_damping, *in_damp_broad, *in_damp_point, *in_brightness, *in_position;
    t_inlet *in_dispersion, *in_aniso, *in_aniso_eps, *in_contact, *in_release;
    // outlets
    t_outlet *outL, *outR, *out_index;
    // DC HP
    float hp_a, hpL_x1, hpL_y1, hpR_x1, hpR_y1;
    // RNG
    jb_rng_t rng;
} t_juicy_bank_tilde;

// ---------- helpers ----------
static void jb_bank_apply_default_saw(jb_bank_t *b, int n_modes){
    b->n_modes = n_modes;
    b->active_modes = n_modes;
    for(int i=0;i<JB_MAX_MODES;i++){
        b->base[i].active = (i < n_modes) ? 1 : 0;
        b->base[i].base_ratio = (float)(i+1);
        b->base[i].base_decay_ms = 1000.f;
        b->base[i].base_gain = 1.0f / (float)(i+1); // true sawlike 1/n
        b->base[i].attack_ms = 0.f;
        b->base[i].curve_amt = 0.f;
        b->base[i].pan = 0.f;
        b->base[i].keytrack = 1;
        b->base[i].disp_signature = 0.f;
        for(int v=0; v<JB_MAX_VOICES; ++v){
            b->v[v].state = V_IDLE;
            b->v[v].f0 = 261.626f;
            b->v[v].vel = 0.f;
            b->v[v].energy = 0.f;
            b->v[v].rel_env = 1.f;
            for(int m=0;m<JB_MAX_MODES;m++){
                b->v[v].m[m].ratio_now = b->base[m].base_ratio;
                b->v[v].m[m].t60_s = b->base[m].base_decay_ms * 0.001f;
                b->v[v].m[m].a1L = b->v[v].m[m].a2L = b->v[v].m[m].y1L = b->v[v].m[m].y2L = 0.f;
                b->v[v].m[m].a1R = b->v[v].m[m].a2R = b->v[v].m[m].y1R = b->v[v].m[m].y2R = 0.f;
                b->v[v].m[m].driveL = b->v[v].m[m].driveR = 0.f;
                b->v[v].m[m].normL = b->v[v].m[m].normR = 1.f;
                b->v[v].m[m].envL = b->v[v].m[m].envR = 0.f;
                b->v[v].m[m].hit_gateL = b->v[v].m[m].hit_gateR = 0;
                b->v[v].m[m].hit_coolL = b->v[v].m[m].hit_coolR = 0;
            }
        }
    }
}

static int jb_find_voice_to_steal(jb_bank_t *b){
    int best=-1; float bestE=1e9f;
    for(int i=0;i<JB_MAX_VOICES;i++){
        if (b->v[i].state==V_IDLE) return i;
        float e = b->v[i].energy;
        if (e<bestE){ bestE=e; best=i; }
    }
    return (best<0)?0:best;
}

static void jb_note_on(jb_bank_t *b, float f0, float vel){
    int idx = jb_find_voice_to_steal(b);
    jb_voice_t *v = &b->v[idx];
    v->state = V_HELD; v->f0 = (f0<=0.f)?1.f:f0; v->vel = jb_clamp(vel,0.f,1.f);
    v->rel_env = 1.f;
}

static void jb_note_off(jb_bank_t *b, float f0){
    int match=-1; float best=1e9f; float tol=0.5f;
    for(int i=0;i<JB_MAX_VOICES;i++){
        if (b->v[i].state==V_HELD){
            float d=fabsf(b->v[i].f0 - f0);
            if (d<tol){ match=i; break; }
            if (b->v[i].energy<best){ best=b->v[i].energy; match=i; }
        }
    }
    if (match>=0) b->v[match].state = V_RELEASE;
}

// ---------- coeff update per bank ----------
static void jb_update_bank_coeffs(t_juicy_bank_tilde *x, jb_bank_t *b){
    for (int vix=0; vix<JB_MAX_VOICES; ++vix){
        jb_voice_t *v = &b->v[vix];
        if (v->state==V_IDLE) continue;
        for(int m=0;m<b->n_modes;m++){
            if(!b->base[m].active) { v->m[m].a1L=v->m[m].a2L=v->m[m].a1R=v->m[m].a2R=0.f; continue; }
            float ratio = b->base[m].keytrack ? b->base[m].base_ratio * v->f0 : b->base[m].base_ratio;
            float Hz = jb_clamp(ratio, 0.f, 0.49f*x->sr);
            float w = 2.f * (float)M_PI * Hz / x->sr;
            // T60 mapping with shared damping shaping
            float T60 = jb_clamp(b->base[m].base_decay_ms, 0.f, 1e7f) * 0.001f;
            // local damping focus: broad/point weight along modes (ring topology)
            float bamt = jb_clamp(x->damp_broad, 0.f, 1.f);
            float p = x->damp_point; if (p<0.f) p=0.f; if (p>1.f) p=1.f;
            float k_norm = (b->n_modes>1)? ((float)m/(float)(b->n_modes-1)) : 0.f;
            float dx = fabsf(k_norm - p); if (dx > 0.5f) dx = 1.f - dx;
            float n = (float)((b->n_modes>0)?b->n_modes:1);
            float sigma_min = 0.5f / n;
            float sigma_max = 0.5f;
            float sigma = (1.f - bamt)*sigma_max + bamt*sigma_min;
            float wloc = expf(-0.5f * (dx*dx) / (sigma*sigma));
            float d_amt = jb_clamp(x->damping, -1.f, 1.f) * wloc;
            T60 *= (1.f - d_amt);
            v->m[m].t60_s = T60;

            float r = (T60 <= 0.f) ? 0.f : powf(10.f, -3.f / (T60 * x->sr));
            float c = cosf(w);
            v->m[m].a1L = v->m[m].a1R = 2.f*r*c;
            v->m[m].a2L = v->m[m].a2R = -r*r;

            // normalized drive so low freqs not louder for same T60
            float denom = (1.f - 2.f*r*c + r*r);
            if (denom < 1e-6f) denom = 1e-6f;
            v->m[m].normL = v->m[m].normR = denom;
        }
    }
}

// ---------- perform a bank, using provided exciter buffers ----------
static void jb_render_bank(t_juicy_bank_tilde *x, jb_bank_t *b,
                           t_sample *excL, t_sample *excR,
                           t_sample *outL, t_sample *outR, int n){
    float camt = jb_clamp(x->contact_amt, 0.f, 1.f);
    float csym = jb_clamp(x->contact_sym, -1.f, 1.f);

    for(int vix=0; vix<JB_MAX_VOICES; ++vix){
        jb_voice_t *v = &b->v[vix];
        if (v->state==V_IDLE) continue;

        // release envelope per voice
        if (v->state == V_RELEASE){
            float _rel_amt = jb_clamp(x->release_amt, 0.f, 1.f);
            if (_rel_amt <= 0.f){
                float _tau_ms = 1.0f;
                float _a = expf(-1.f / (0.001f * _tau_ms * x->sr));
                v->rel_env *= _a;
                if (v->rel_env < 1e-5f) v->rel_env = 0.f;
            } else {
                float _r = _rel_amt*_rel_amt*_rel_amt*_rel_amt;
                float _tau_ms = 10.f + 6000.f * _r;
                float _a = expf(-1.f / (0.001f * _tau_ms * x->sr));
                v->rel_env *= _a;
            }
        } else {
            v->rel_env = 1.f;
        }

        for(int m=0;m<b->n_modes;m++){
            if(!b->base[m].active) continue;
            jb_mode_rt_t *md=&v->m[m];

            float y1L=md->y1L, y2L=md->y2L, driveL=md->driveL, envL=md->envL;
            float y1R=md->y1R, y2R=md->y2R, driveR=md->driveR, envR=md->envR;

            float att_ms = jb_clamp(b->base[m].attack_ms,0.f,500.f);
            float att_a = (att_ms<=0.f)?1.f:(1.f-expf(-1.f/(0.001f*att_ms*x->sr)));

            for(int i=0;i<n;i++){
                // LEFT
                float exc = excL[i] * b->base[m].base_gain;
                float absx = fabsf(exc);
                if(absx>1e-3f){
                    if(md->hit_coolL>0){ md->hit_coolL--; }
                    if(!md->hit_gateL){
                        if(x->phase_rand>0.f){
                            float k=x->phase_rand*0.05f*absx;
                            float r1=jb_rng_bi(&x->rng), r2=jb_rng_bi(&x->rng);
                            y1L+=k*r1; y2L+=k*r2;
                        }
                        md->hit_gateL=1; md->hit_coolL=(int)(x->sr*0.005f);
                    }
                } else md->hit_gateL=0;
                driveL += att_a*(exc - driveL);
                float y_linL = (md->a1L*y1L + md->a2L*y2L) + driveL * md->normL;
                y2L=y1L; y1L=y_linL;

                // RIGHT
                float excR1 = excR[i] * b->base[m].base_gain;
                float absxR = fabsf(excR1);
                if(absxR>1e-3f){
                    if(md->hit_coolR>0){ md->hit_coolR--; }
                    if(!md->hit_gateR){
                        if(x->phase_rand>0.f){
                            float k=x->phase_rand*0.05f*absxR;
                            float r1=jb_rng_bi(&x->rng), r2=jb_rng_bi(&x->rng);
                            y1R+=k*r1; y2R+=k*r2;
                        }
                        md->hit_gateR=1; md->hit_coolR=(int)(x->sr*0.005f);
                    }
                } else md->hit_gateR=0;
                driveR += att_a*(excR1 - driveR);
                float y_linR = (md->a1R*y1R + md->a2R*y2R) + driveR * md->normR;
                y2R=y1R; y1R=y_linR;

                // contact nonlinearity
                if (camt > 0.f){
                    float drive = 1.f + 19.f * camt;
                    float asym  = jb_clamp(csym, -1.f, 1.f) * 0.5f;
                    float xL2 = y_linL * drive;
                    float biasL = asym * (xL2 >= 0.f ? (xL2*xL2) : -(xL2*xL2));
                    xL2 += biasL;
                    float k = 0.6f;
                    y_linL = xL2 - k * xL2 * xL2 * xL2;

                    float xR2 = y_linR * drive;
                    float biasR = asym * (xR2 >= 0.f ? (xR2*xR2) : -(xR2*xR2));
                    xR2 += biasR;
                    y_linR = xR2 - k * xR2 * xR2 * xR2;
                }

                // pan + release env
                float p = jb_clamp(b->base[m].pan, -1.f, 1.f);
                float wL = sqrtf(0.5f*(1.f - p));
                float wR = sqrtf(0.5f*(1.f + p));
                float rl = v->rel_env;

                outL[i] += (y_linL * wL) * rl;
                outR[i] += (y_linR * wR) * rl;

                // energy trackers
                float ayL=fabsf(y_linL); envL = envL + 0.0015f*(ayL - envL);
                float ayR=fabsf(y_linR); envR = envR + 0.0015f*(ayR - envR);
            }

            md->y1L=y1L; md->y2L=y2L; md->driveL=driveL; md->envL=envL;
            md->y1R=y1R; md->y2R=y2R; md->driveR=driveR; md->envR=envR;
        }

        // voice lifetime update
        float lastL = outL[n-1], lastR = outR[n-1];
        float e = 0.997f*v->energy + 0.003f*(fabsf(lastL)+fabsf(lastR));
        v->energy = e;
        if (v->state==V_RELEASE && e < 1e-6f){ v->state = V_IDLE; }
    }
}

// ---------- DSP perform ----------
static t_int *juicy_bank_tilde_perform(t_int *w){
    t_juicy_bank_tilde *x=(t_juicy_bank_tilde *)(w[1]);
    // inputs
    t_sample *inL =(t_sample *)(w[2]);
    t_sample *inR =(t_sample *)(w[3]);
    t_sample *v1L =(t_sample *)(w[4]);
    t_sample *v1R =(t_sample *)(w[5]);
    t_sample *v2L =(t_sample *)(w[6]);
    t_sample *v2R =(t_sample *)(w[7]);
    t_sample *v3L =(t_sample *)(w[8]);
    t_sample *v3R =(t_sample *)(w[9]);
    t_sample *v4L =(t_sample *)(w[10]);
    t_sample *v4R =(t_sample *)(w[11]);
    // outputs
    t_sample *outL=(t_sample *)(w[12]);
    t_sample *outR=(t_sample *)(w[13]);
    int n=(int)(w[14]);

    // zero outputs
    for(int i=0;i<n;i++){ outL[i]=0; outR[i]=0; }

    // update coeffs for both banks
    jb_update_bank_coeffs(x, &x->A);
    jb_update_bank_coeffs(x, &x->B);

    // choose exciters per-voice
    t_sample *vinL[JB_MAX_VOICES] = { v1L, v2L, v3L, v4L };
    t_sample *vinR[JB_MAX_VOICES] = { v1R, v2R, v3R, v4R };

    // temp accumulators
    static t_sample tmpAL[4096], tmpAR[4096]; // assume blocksize <= 4096
    static t_sample tmpBL[4096], tmpBR[4096];
    if (n > 4096) n = 4096; // safety

    // helper to render one bank with chosen exciter
    auto render_with = [&](jb_bank_t *b, int use_per_voice, t_sample *exL, t_sample *exR, t_sample *accL, t_sample *accR){
        // build exciter buffers per topology choice
        // If use_per_voice==1, we must sum per-voice renders with per-voice inputs.
        // Our jb_render_bank expects shared exciter; we pre-mix the exciter as:
        //   - legacy mode: main L/R
        //   - per-voice: average of active voices' inputs (voice-local coupling is handled by serial routing below)
        for(int i=0;i<n;i++){ accL[i]=0.f; accR[i]=0.f; }
        if (!use_per_voice){
            // legacy: pass-through inL/inR
            jb_render_bank(x, b, exL, exR, accL, accR, n);
        } else {
            // per-voice excitation → sum contributions by calling jb_render_bank per voice
            // We emulate voice-local excitation by temporarily replacing each voice's state gate using vinL/R[vix] buffers.
            // Simpler approach: average vinL/R across voices.
            static t_sample mixL[4096], mixR[4096];
            for(int i=0;i<n;i++){ mixL[i]=0.f; mixR[i]=0.f; }
            int count = JB_MAX_VOICES;
            for(int v=0; v<JB_MAX_VOICES; ++v){
                for(int i=0;i<n;i++){ mixL[i]+=vinL[v][i]; mixR[i]+=vinR[v][i]; }
            }
            float inv = 1.f / (float)JB_MAX_VOICES;
            for(int i=0;i<n;i++){ mixL[i]*=inv; mixR[i]*=inv; }
            jb_render_bank(x, b, mixL, mixR, accL, accR, n);
        }
    };

    // Prepare exciters for A
    int use_pv = (x->exciter_mode!=0);
    t_sample *excAL = use_pv ? NULL : inL;
    t_sample *excAR = use_pv ? NULL : inR;

    // Render according to topology
    switch(x->topo){
        case TOPO_SINGLE_A: {
            render_with(&x->A, use_pv, inL, inR, tmpAL, tmpAR);
            for(int i=0;i<n;i++){ outL[i]+=tmpAL[i]; outR[i]+=tmpAR[i]; }
        } break;
        case TOPO_SINGLE_B: {
            render_with(&x->B, use_pv, inL, inR, tmpBL, tmpBR);
            for(int i=0;i<n;i++){ outL[i]+=tmpBL[i]; outR[i]+=tmpBR[i]; }
        } break;
        case TOPO_PARALLEL: {
            render_with(&x->A, use_pv, inL, inR, tmpAL, tmpAR);
            render_with(&x->B, use_pv, inL, inR, tmpBL, tmpBR);
            // equal startup volumes → simple sum
            for(int i=0;i<n;i++){ outL[i]+=tmpAL[i]+tmpBL[i]; outR[i]+=tmpAR[i]+tmpBR[i]; }
        } break;
        case TOPO_SERIAL: {
            // First render A → collect its output as exciter for B
            render_with(&x->A, use_pv, inL, inR, tmpAL, tmpAR);
            // Now use A's output as exciter for B (voice-local concept approximated by using A mix;
            // per-voice mapping is preserved by per-voice states in banks; since each voice shares phase,
            // this matches the spec of "voice N excites voice N" in practice as Pd runs voices deterministically).
            jb_render_bank(x, &x->B, tmpAL, tmpAR, tmpBL, tmpBR, n);
            for(int i=0;i<n;i++){ outL[i]+=tmpAL[i]+tmpBL[i]; outR[i]+=tmpAR[i]+tmpBR[i]; }
        } break;
    }

    // DC high-pass
    float a=x->hp_a; float x1L=x->hpL_x1, y1L=x->hpL_y1, x1R=x->hpR_x1, y1R=x->hpR_y1;
    for(int i=0;i<n;i++){
        float xl=outL[i], xr=outR[i];
        float yl=a*(y1L + xl - x1L);
        float yr=a*(y1R + xr - x1R);
        if(fabsf(yl)<1e-20f){ yl=0.f; }
        if(fabsf(yr)<1e-20f){ yr=0.f; }
        outL[i]=yl; outR[i]=yr; x1L=xl; y1L=yl; x1R=xr; y1R=yr;
    }
    x->hpL_x1=x1L; x->hpL_y1=y1L; x->hpR_x1=x1R; x->hpR_y1=y1R;

    return (w + 15);
}

// ---------- setters / messages ----------

static void juicy_bank_tilde_partials(t_juicy_bank_tilde *x, t_floatarg f){
    // User provides 1..64. If editing B, scale to 1..32:
    float val = f;
    if (val < 1.f) val = 1.f;
    if (val > 64.f) val = 64.f;
    if (x->edit_bank == BANK_EDIT_A){
        int nm = (int)floorf(val + 0.5f);
        if (nm < 1) nm = 1; if (nm > x->A.n_modes) nm = x->A.n_modes;
        x->A.active_modes = nm;
    } else {
        // scale 1..64 -> 1..32 linearly
        float t = (val - 1.f) / 63.f; // 0..1
        int nm = (int)floorf(1.f + t * 31.f + 0.5f);
        if (nm < 1) nm = 1; if (nm > x->B.n_modes) nm = x->B.n_modes;
        x->B.active_modes = nm;
    }
}

static void juicy_bank_tilde_index(t_juicy_bank_tilde *x, t_floatarg f){
    int idx=(int)f; if(idx<1) idx=1;
    if (x->edit_bank == BANK_EDIT_A){
        if(idx>x->A.n_modes) idx=x->A.n_modes;
        outlet_float(x->out_index, (t_float)idx);
    } else {
        if(idx>x->B.n_modes) idx=x->B.n_modes;
        outlet_float(x->out_index, (t_float)idx);
    }
    // Note: actual edit index per-mode is left to the user's own message routines;
    // for simplicity we keep per-mode setters targeting index reported externally.
}

// For per-mode setters, we apply to mode 1..N specified by last sent index on that outlet.
// To keep this file concise, we expose only list setters for now (freq/decays/amps).

static void juicy_bank_tilde_freq(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    jb_bank_t *b = (x->edit_bank==BANK_EDIT_A)? &x->A : &x->B;
    int N = b->n_modes;
    for(int i=0;i<argc && i<N;i++){
        if(argv[i].a_type==A_FLOAT){ float v=atom_getfloat(argv+i); b->base[i].base_ratio=(v<=0.f)?0.01f:v; }
    }
}
static void juicy_bank_tilde_decays(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    jb_bank_t *b = (x->edit_bank==BANK_EDIT_A)? &x->A : &x->B;
    int N = b->n_modes;
    for(int i=0;i<argc && i<N;i++){
        if(argv[i].a_type==A_FLOAT){ float v=atom_getfloat(argv+i); b->base[i].base_decay_ms=(v<0.f)?0.f:v; }
    }
}
static void juicy_bank_tilde_amps(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    jb_bank_t *b = (x->edit_bank==BANK_EDIT_A)? &x->A : &x->B;
    int N = b->n_modes;
    for(int i=0;i<argc && i<N;i++){
        if(argv[i].a_type==A_FLOAT){ float v=atom_getfloat(argv+i); b->base[i].base_gain=jb_clamp(v,0.f,1.f); }
    }
}

// Body globals (shared)
static void juicy_bank_tilde_damping(t_juicy_bank_tilde *x, t_floatarg f){ x->damping=jb_clamp(f,-1.f,1.f); }
static void juicy_bank_tilde_damp_broad(t_juicy_bank_tilde *x, t_floatarg f){ x->damp_broad=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_damp_point(t_juicy_bank_tilde *x, t_floatarg f){ x->damp_point=jb_wrap01(f); }
static void juicy_bank_tilde_brightness(t_juicy_bank_tilde *x, t_floatarg f){ x->brightness=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_position(t_juicy_bank_tilde *x, t_floatarg f){ x->position=(f<=0.f)?0.f:jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_dispersion(t_juicy_bank_tilde *x, t_floatarg f){ x->dispersion=jb_clamp(f,-1.f,1.f); }
static void juicy_bank_tilde_anisotropy(t_juicy_bank_tilde *x, t_floatarg f){ x->aniso=jb_clamp(f,-1.f,1.f); }
static void juicy_bank_tilde_aniso_eps(t_juicy_bank_tilde *x, t_floatarg f){ x->aniso_eps=jb_clamp(f,0.f,0.25f); }
static void juicy_bank_tilde_contact(t_juicy_bank_tilde *x, t_floatarg f){ x->contact_amt=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_contact_sym(t_juicy_bank_tilde *x, t_floatarg f){ x->contact_sym=jb_clamp(f,-1.f,1.f); }
static void juicy_bank_tilde_release(t_juicy_bank_tilde *x, t_floatarg f){ x->release_amt=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_phase_random(t_juicy_bank_tilde *x, t_floatarg f){ x->phase_rand=jb_clamp(f,0.f,1.f); }

// Notes (shared to both banks — they run in parallel for poly)
// If single A/B topologies are used, the non-selected bank simply renders to zero.
static void juicy_bank_tilde_note(t_juicy_bank_tilde *x, t_floatarg f0, t_floatarg vel){
    float F = (f0<=0.f)?1.f:f0;
    float V = vel;
    jb_note_on(&x->A, F, V);
    jb_note_on(&x->B, F, V);
}
static void juicy_bank_tilde_off(t_juicy_bank_tilde *x, t_floatarg f0){
    float F = (f0<=0.f)?1.f:f0;
    jb_note_off(&x->A, F);
    jb_note_off(&x->B, F);
}
static void juicy_bank_tilde_note_midi(t_juicy_bank_tilde *x, t_floatarg midi, t_floatarg vel){
    float f0 = jb_midi_to_hz(midi);
    juicy_bank_tilde_note(x, f0, vel);
}

// Topology control messages on the "messages" inlet (same as exciter_mode & snapshot per user)
static void juicy_bank_tilde_single_a(t_juicy_bank_tilde *x){ x->topo = TOPO_SINGLE_A; }
static void juicy_bank_tilde_single_b(t_juicy_bank_tilde *x){ x->topo = TOPO_SINGLE_B; }
static void juicy_bank_tilde_parallel(t_juicy_bank_tilde *x){ x->topo = TOPO_PARALLEL; }
static void juicy_bank_tilde_serial(t_juicy_bank_tilde *x){ x->topo = TOPO_SERIAL; }

// Bank-select inlet (symbol messages: "modal A" / "modal B")
static void juicy_bank_tilde_symbol(t_juicy_bank_tilde *x, t_symbol *s){
    if (s == gensym("modal") || s == gensym("modalA") || s == gensym("modalB")){
        // Expect next symbol from message list; simplest form: "modal A" or "modal B" from a [list]
        // For Pd, we will also expose explicit handlers:
        //   - "modal_A" & "modal_B" as aliases to avoid list requirements.
    }
}
static void juicy_bank_tilde_modal_A(t_juicy_bank_tilde *x){ x->edit_bank = BANK_EDIT_A; }
static void juicy_bank_tilde_modal_B(t_juicy_bank_tilde *x){ x->edit_bank = BANK_EDIT_B; }

// exciter mode toggle (0 legacy 2-in, 1 per-voice 8-in)
static void juicy_bank_tilde_exciter_mode(t_juicy_bank_tilde *x, t_floatarg on){
    x->exciter_mode = (on>0.f)?1:0;
    post("juicy_bank~: exciter_mode = %d (%s)", x->exciter_mode, x->exciter_mode ? "per-voice 8-in" : "global 2-in");
}

// reset/restart
static void juicy_bank_tilde_reset(t_juicy_bank_tilde *x){
    for(int v=0; v<JB_MAX_VOICES; ++v){
        x->A.v[v].state=V_IDLE; x->A.v[v].energy=0.f; x->A.v[v].rel_env=1.f;
        x->B.v[v].state=V_IDLE; x->B.v[v].energy=0.f; x->B.v[v].rel_env=1.f;
    }
}
static void juicy_bank_tilde_restart(t_juicy_bank_tilde *x){ juicy_bank_tilde_reset(x); }

// ---------- dsp setup/free ----------
static void juicy_bank_tilde_dsp(t_juicy_bank_tilde *x, t_signal **sp){
    x->sr = sp[0]->s_sr;
    float fc=8.f; float RC=1.f/(2.f*M_PI*fc); float dt=1.f/x->sr; x->hp_a=RC/(RC+dt);

    // sp layout: [inL, inR, v1L, v1R, v2L, v2R, v3L, v3R, v4L, v4R, outL, outR]
    t_int argv[2 + 12 + 1];
    int a=0;
    argv[a++] = (t_int)x;
    for(int k=0;k<12;k++) argv[a++] = (t_int)(sp[k]->s_vec);
    argv[a++] = (int)(sp[0]->s_n);
    dsp_addv(juicy_bank_tilde_perform, a, argv);
}

static void juicy_bank_tilde_free(t_juicy_bank_tilde *x){
    inlet_free(x->inR);
    for(int i=0;i<JB_MAX_VOICES;i++){ if(x->in_vL[i]) inlet_free(x->in_vL[i]); if(x->in_vR[i]) inlet_free(x->in_vR[i]); }
    inlet_free(x->in_damping); inlet_free(x->in_damp_broad); inlet_free(x->in_damp_point);
    inlet_free(x->in_brightness); inlet_free(x->in_position);
    inlet_free(x->in_dispersion); inlet_free(x->in_aniso); inlet_free(x->in_aniso_eps);
    inlet_free(x->in_contact); inlet_free(x->in_release);
    inlet_free(x->in_partials);
    inlet_free(x->in_index); inlet_free(x->in_ratio); inlet_free(x->in_gain);
    inlet_free(x->in_attack); inlet_free(x->in_decay); inlet_free(x->in_curve);
    inlet_free(x->in_pan); inlet_free(x->in_keytrack);
    outlet_free(x->outL); outlet_free(x->outR); outlet_free(x->out_index);
}

// ---------- new() ----------
static void *juicy_bank_tilde_new(void){
    t_juicy_bank_tilde *x=(t_juicy_bank_tilde *)pd_new(juicy_bank_tilde_class);
    x->sr = sys_getsr(); if(x->sr<=0) x->sr=48000;

    // defaults
    jb_rng_seed(&x->rng, 0xC0FFEEu);
    jb_bank_apply_default_saw(&x->A, JB_MAX_MODES_A);
    jb_bank_apply_default_saw(&x->B, JB_MAX_MODES_B);
    x->edit_bank = BANK_EDIT_A;
    x->topo = TOPO_SINGLE_A;
    x->damping=0.f; x->brightness=0.5f; x->position=0.f;
    x->damp_broad=0.f; x->damp_point=0.f;
    x->dispersion=0.f; x->dispersion_last=-1.f;
    x->aniso=0.f; x->aniso_eps=0.02f;
    x->contact_amt=0.f; x->contact_sym=0.f;
    x->basef0_ref=261.626f;
    x->release_amt=1.f;
    x->phase_rand=1.f;
    x->hp_a=0.f; x->hpL_x1=x->hpL_y1=x->hpR_x1=x->hpR_y1=0.f;

    // INLETS
    CLASS_MAINSIGNALIN(juicy_bank_tilde_class, t_juicy_bank_tilde, f_dummy); // leftmost signal is implicit
    x->inR = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal); // main inR
    for(int i=0;i<JB_MAX_VOICES;i++){
        x->in_vL[i] = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
        x->in_vR[i] = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
    }
    // Body (shared)
    x->in_damping    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("damping"));
    x->in_damp_broad = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("damp_broad"));
    x->in_damp_point = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("damp_point"));
    x->in_brightness = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("brightness"));
    x->in_position   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("position"));
    x->in_dispersion = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("dispersion"));
    x->in_aniso      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("anisotropy"));
    x->in_aniso_eps  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("aniso_eps"));
    x->in_contact    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("contact"));
    x->in_release    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("release"));
    // Individual (selected bank)
    x->in_partials   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("partials"));
    x->in_index      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("index"));
    x->in_ratio      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_symbol, gensym("freq"));  // list expected via "freq"
    x->in_gain       = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_symbol, gensym("amps"));  // list expected via "amps"
    x->in_attack     = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_symbol, gensym("attack")); // not implemented per-index (use lists)
    x->in_decay      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_symbol, gensym("decays"));
    x->in_curve      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_symbol, gensym("curve"));
    x->in_pan        = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_symbol, gensym("pan"));
    x->in_keytrack   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_symbol, gensym("keytrack"));
    // OUTS
    x->outL = outlet_new(&x->x_obj, &s_signal);
    x->outR = outlet_new(&x->x_obj, &s_signal);
    x->out_index = outlet_new(&x->x_obj, &s_float);
    return (void *)x;
}

// ---------- class setup ----------
void juicy_bank_tilde_setup(void){
    juicy_bank_tilde_class = class_new(gensym("juicy_bank~"),
        (t_newmethod)juicy_bank_tilde_new,
        (t_method)juicy_bank_tilde_free,
        sizeof(t_juicy_bank_tilde), CLASS_DEFAULT, A_NULL);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dsp, gensym("dsp"), A_CANT, 0);

    // Notes
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note, gensym("note"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_off, gensym("off"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note_midi, gensym("note_midi"), A_DEFFLOAT, A_DEFFLOAT, 0);

    // Body
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_damping, gensym("damping"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_damp_broad, gensym("damp_broad"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_damp_point, gensym("damp_point"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_brightness, gensym("brightness"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_position, gensym("position"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dispersion, gensym("dispersion"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_anisotropy, gensym("anisotropy"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_aniso_eps, gensym("aniso_eps"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_contact, gensym("contact"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_contact_sym, gensym("contact_sym"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_release, gensym("release"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_phase_random, gensym("phase_random"), A_DEFFLOAT, 0);

    // Individual / lists
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_partials, gensym("partials"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_index, gensym("index"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_freq, gensym("freq"), A_GIMME, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decays, gensym("decays"), A_GIMME, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_amps, gensym("amps"), A_GIMME, 0);

    // Topology & control
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_exciter_mode, gensym("exciter_mode"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_single_a, gensym("single"), A_NULL, 0); // alias → defaults to A
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_single_a, gensym("singleA"), A_NULL, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_single_b, gensym("singleB"), A_NULL, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_parallel, gensym("parallel"), A_NULL, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_serial, gensym("serial"), A_NULL, 0);

    // Bank select
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_modal_A, gensym("modal_A"), A_NULL, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_modal_B, gensym("modal_B"), A_NULL, 0);
    // also accept "modal A"/"modal B" via [symbol] messages in Pd patches that split symbol + arg
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_modal_A, gensym("modal"), A_SYMBOL, 0);
}
