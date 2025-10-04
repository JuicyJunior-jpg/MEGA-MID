// juicy_bank~ — modal resonator bank (V4.9)
// 4-voice poly, true stereo banks, Behavior + Body + Individual inlets.
// NEW (V4.9):
//   • **Per-voice exciter inputs**: 8 extra signal inlets [v1_L v1_R v2_L v2_R v3_L v3_R v4_L v4_R]
//     Use message **exciter_mode 1** to drive each voice from its own stereo input (pair comes from juicy_exciter~).
//     Default **exciter_mode 0** keeps legacy 2-in global excitation (stereo) feeding held voices.
//   • Backward compatible: if you don't send exciter_mode 1, patch behaves exactly like older builds.
//
// INLET GROUPS (left → right):
//  • SIGNAL (10 total): inL, inR, v1_L, v1_R, v2_L, v2_R, v3_L, v3_R, v4_L, v4_R
//  • BEHAVIOR (7): stiffen, shortscle, linger, tilt, bite, bloom, crossring
//  • BODY (7):     damping, brightness, position, density, dispersion, anisotropy, contact
//  • INDIVIDUAL (8, per-mode via index): index, ratio, gain, attack, decya, curve, pan, keytrack
//
// Voice-addressed poly for Pd [poly]:
//  • note_poly <v> <Hz> <vel>  • note_poly_midi <v> <midinote> <vel>  • off_poly <v>
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

#define JB_MAX_MODES   64
#define JB_MAX_VOICES   4

// ---------- utils ----------
static inline float jb_clamp(float x, float lo, float hi){ return (x<lo)?lo:((x>hi)?hi:x); }
typedef struct { unsigned int s; } jb_rng_t;
static inline void jb_rng_seed(jb_rng_t *r, unsigned int s){ if(!s) s=1; r->s = s; }
static inline unsigned int jb_rng_u32(jb_rng_t *r){ unsigned int x = r->s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; r->s = x; return x; }
static inline float jb_rng_uni(jb_rng_t *r){ return (jb_rng_u32(r) >> 8) * (1.0f/16777216.0f); }
static inline float jb_rng_bi(jb_rng_t *r){ return 2.f * jb_rng_uni(r) - 1.f; }
static int jb_is_near_integer(float x, float eps){ float n=roundf(x); return fabsf(x-n)<=eps; }
static inline float jb_midi_to_hz(float n){ return 440.f * powf(2.f, (n-69.f)/12.f); }

typedef enum { DENSITY_PIVOT=0, DENSITY_INDIV=1 } jb_density_mode;

typedef struct {
    // base params (shared template per mode)
    float base_ratio, base_decay_ms, base_gain;
    float attack_ms, curve_amt, pan;
    int   active;
    int   keytrack; // 1 = track f0 (ratio), 0 = absolute Hz

    // signatures (random)
    float disp_signature;
    float micro_sig;
} jb_mode_base_t;

typedef struct {
    // runtime per-mode
    float ratio_now, decay_ms_now, gain_now;
    float t60_s, decay_u;

    // per-ear per-hit randomizations
    float md_hit_offsetL, md_hit_offsetR;   // micro detune offsets
    float bw_hit_ratioL, bw_hit_ratioR;     // twin detune ratios

    // LEFT states
    float a1L,a2L, y1L,y2L, a1bL,a2bL, y1bL,y2bL, envL, y_pre_lastL;
    // RIGHT states
    float a1R,a2R, y1R,y2R, a1bR,a2bR, y1bR,y2bR, envR, y_pre_lastR;

    // drive/hit
    float driveL, driveR;
    int   hit_gateL, hit_coolL, hit_gateR, hit_coolR;
} jb_mode_rt_t;

typedef enum { V_IDLE=0, V_HELD=1, V_RELEASE=2 } jb_vstate;

typedef struct {
    jb_vstate state;
    float f0, vel, energy;

    // projected behavior (per voice)
    float pitch_x;
    float brightness_v;
    float bandwidth_v;
    float decay_pitch_mul;
    float decay_vel_mul;
    float stiffen_add;

    // sympathetic multipliers
    float cr_gain_mul[JB_MAX_MODES];
    float cr_decay_mul[JB_MAX_MODES];

    // dispersion morph targets
    float disp_offset[JB_MAX_MODES];
    float disp_target[JB_MAX_MODES];

    // runtime per-mode
    jb_mode_rt_t m[JB_MAX_MODES];
} jb_voice_t;

// ---------- the object ----------
static t_class *juicy_bank_tilde_class;

typedef struct _juicy_bank_tilde {
    t_object  x_obj; t_float f_dummy; t_float sr;

    int n_modes;
    jb_mode_base_t base[JB_MAX_MODES];

    // BODY globals
    float damping, brightness, position;
    float density_amt; jb_density_mode density_mode;
    float dispersion, dispersion_last;
    float aniso, aniso_eps;
    float contact_amt, contact_sym;

    // realism/misc
    float phase_rand; int phase_debug;
    float bandwidth;        // base for Bloom
    float micro_detune;     // base for micro detune
    float basef0_ref;
    // BEHAVIOR (reduced)

    // BODY globals
static void juicy_bank_tilde_damping(t_juicy_bank_tilde *x, t_floatarg f){ x->damping=jb_clamp(f,-1.f,1.f); }
static void juicy_bank_tilde_brightness(t_juicy_bank_tilde *x, t_floatarg f){ x->brightness=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_position(t_juicy_bank_tilde *x, t_floatarg f){ x->position=(f<=0.f)?0.f:jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_density(t_juicy_bank_tilde *x, t_floatarg f){ x->density_amt=jb_clamp(f,-1.f,1.f); }
static void juicy_bank_tilde_density_pivot(t_juicy_bank_tilde *x){ x->density_mode=DENSITY_PIVOT; }
static void juicy_bank_tilde_density_individual(t_juicy_bank_tilde *x){ x->density_mode=DENSITY_INDIV; }
static void juicy_bank_tilde_anisotropy(t_juicy_bank_tilde *x, t_floatarg f){ x->aniso=jb_clamp(f,-1.f,1.f); }
static void juicy_bank_tilde_aniso_eps(t_juicy_bank_tilde *x, t_floatarg f){ x->aniso_eps=jb_clamp(f,0.f,0.25f); }
static void juicy_bank_tilde_contact(t_juicy_bank_tilde *x, t_floatarg f){ x->contact_amt=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_contact_sym(t_juicy_bank_tilde *x, t_floatarg f){ x->contact_sym=jb_clamp(f,-1.f,1.f); }

// realism & misc
static void juicy_bank_tilde_phase_random(t_juicy_bank_tilde *x, t_floatarg f){ x->phase_rand=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_phase_debug(t_juicy_bank_tilde *x, t_floatarg on){ x->phase_debug=(on>0.f)?1:0; }
static void juicy_bank_tilde_bandwidth(t_juicy_bank_tilde *x, t_floatarg f){ x->bandwidth=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_micro_detune(t_juicy_bank_tilde *x, t_floatarg f){ x->micro_detune=jb_clamp(f,0.f,1.f); }

// dispersion & seeds
static void juicy_bank_tilde_dispersion(t_juicy_bank_tilde *x, t_floatarg f){
    float v=jb_clamp(f,0.f,1.f);
    if(x->dispersion_last<0.f || fabsf(v-x->dispersion_last)>1e-6f){
        for(int i=0;i<x->n_modes;i++){ x->base[i].disp_signature=(i==0)?0.f:jb_rng_bi(&x->rng); }
        x->dispersion_last=v;
    }
    x->dispersion=v;
}
static void juicy_bank_tilde_seed(t_juicy_bank_tilde *x, t_floatarg f){
    jb_rng_seed(&x->rng, (unsigned int)((int)f*2654435761u));
    for(int i=0;i<x->n_modes;i++){
        x->base[i].disp_signature=(i==0)?0.f:jb_rng_bi(&x->rng);
        x->base[i].micro_sig      =(i==0)?0.f:jb_rng_bi(&x->rng);
    }
    x->dispersion_last=x->dispersion;
}
static void juicy_bank_tilde_dispersion_reroll(t_juicy_bank_tilde *x){
    for(int i=0;i<x->n_modes;i++){ x->base[i].disp_signature=(i==0)?0.f:jb_rng_bi(&x->rng); }
    x->dispersion_last=-1.f;
    juicy_bank_tilde_dispersion(x, x->dispersion);
}
    // BEHAVIOR (reduced)

    // BODY
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_damping, gensym("damping"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_brightness, gensym("brightness"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_position, gensym("position"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density, gensym("density"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density_pivot, gensym("density_pivot"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density_individual, gensym("density_individual"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_anisotropy, gensym("anisotropy"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_aniso_eps, gensym("aniso_epsilon"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_contact, gensym("contact"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_contact_sym, gensym("contact_symmetry"), A_DEFFLOAT, 0);

    // INDIVIDUAL (per-mode)
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_index, gensym("index"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_ratio_i, gensym("ratio"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_gain_i, gensym("gain"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_attack_i, gensym("attack"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decay_i, gensym("decay"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decay_i, gensym("decya"), A_DEFFLOAT, 0); // alias
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_curve_i, gensym("curve"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_pan_i, gensym("pan"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_keytrack_i, gensym("keytrack"), A_DEFFLOAT, 0);

    // Lists & misc
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_modes, gensym("modes"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_active, gensym("active"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_freq, gensym("freq"), A_GIMME, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decays, gensym("decays"), A_GIMME, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_amps, gensym("amps"), A_GIMME, 0);

    // dispersion & seeds
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dispersion, gensym("dispersion"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_seed, gensym("seed"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dispersion_reroll, gensym("dispersion_reroll"), 0);

    // realism & misc
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_phase_random, gensym("phase_random"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_phase_debug, gensym("phase_debug"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bandwidth, gensym("bandwidth"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_micro_detune, gensym("micro_detune"), A_DEFFLOAT, 0);

    // notes/poly (non-voice-specific)
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note, gensym("note"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note_midi, gensym("note_midi"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_off, gensym("off"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_voices, gensym("voices"), A_DEFFLOAT, 0);

    // voice-addressed (for [poly])
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note_poly, gensym("note_poly"), A_DEFFLOAT, A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note_poly_midi, gensym("note_poly_midi"), A_DEFFLOAT, A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_off_poly, gensym("off_poly"), A_DEFFLOAT, 0);

    // base & reset
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_basef0, gensym("basef0"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_base_alias, gensym("base"), A_DEFFLOAT, 0);

    // exciter mode
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_exciter_mode, gensym("exciter_mode"), A_DEFFLOAT, 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_reset, gensym("reset"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_restart, gensym("restart"), 0);

    class_sethelpsymbol(juicy_bank_tilde_class, gensym("juicy_bank~"));
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_INIT, gensym("INIT"), 0);
}

void juicy_bank_tilde_setup(void){
    juicy_bank_tilde_class = class_new(gensym("juicy_bank~"),
                           (t_newmethod)juicy_bank_tilde_new,
                           (t_method)juicy_bank_tilde_free,
                           sizeof(t_juicy_bank_tilde), CLASS_DEFAULT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dsp, gensym("dsp"), A_CANT, 0);
    CLASS_MAINSIGNALIN(juicy_bank_tilde_class, t_juicy_bank_tilde, f_dummy);

    // BEHAVIOR    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_crossring, gensym("crossring"), A_DEFFLOAT, 0);

    // BODY
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_damping, gensym("damping"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_brightness, gensym("brightness"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_position, gensym("position"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density, gensym("density"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density_pivot, gensym("density_pivot"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density_individual, gensym("density_individual"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_anisotropy, gensym("anisotropy"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_aniso_eps, gensym("aniso_epsilon"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_contact, gensym("contact"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_contact_sym, gensym("contact_symmetry"), A_DEFFLOAT, 0);

    // INDIVIDUAL (per-mode)
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_index, gensym("index"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_ratio_i, gensym("ratio"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_gain_i, gensym("gain"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_attack_i, gensym("attack"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decay_i, gensym("decay"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decay_i, gensym("decya"), A_DEFFLOAT, 0); // alias
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_curve_i, gensym("curve"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_pan_i, gensym("pan"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_keytrack_i, gensym("keytrack"), A_DEFFLOAT, 0);

    // Lists & misc
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_modes, gensym("modes"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_active, gensym("active"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_freq, gensym("freq"), A_GIMME, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decays, gensym("decays"), A_GIMME, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_amps, gensym("amps"), A_GIMME, 0);

    // dispersion & seeds
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dispersion, gensym("dispersion"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_seed, gensym("seed"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dispersion_reroll, gensym("dispersion_reroll"), 0);

    // realism & misc
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_phase_random, gensym("phase_random"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_phase_debug, gensym("phase_debug"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bandwidth, gensym("bandwidth"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_micro_detune, gensym("micro_detune"), A_DEFFLOAT, 0);

    // notes/poly (non-voice-specific)
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note, gensym("note"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note_midi, gensym("note_midi"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_off, gensym("off"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_voices, gensym("voices"), A_DEFFLOAT, 0);

    // voice-addressed (for [poly])
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note_poly, gensym("note_poly"), A_DEFFLOAT, A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note_poly_midi, gensym("note_poly_midi"), A_DEFFLOAT, A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_off_poly, gensym("off_poly"), A_DEFFLOAT, 0);

    // base & reset
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_basef0, gensym("basef0"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_base_alias, gensym("base"), A_DEFFLOAT, 0);

    // exciter mode
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_exciter_mode, gensym("exciter_mode"), A_DEFFLOAT, 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_reset, gensym("reset"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_restart, gensym("restart"), 0);

    class_sethelpsymbol(juicy_bank_tilde_class, gensym("juicy_bank~"));
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_INIT, gensym("INIT"), 0);
}
