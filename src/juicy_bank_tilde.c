diff --git a/src/juicy_bank_tilde.c b/src/juicy_bank_tilde.c
index 54355069f537e9adfecfed0f4b88a0901d7e859f..c87394266d49e8175f50fed1e52c68b193c9914d 100644
--- a/src/juicy_bank_tilde.c
+++ b/src/juicy_bank_tilde.c
@@ -101,53 +101,59 @@ if (fr < 1.0) fr = 1.0;
 fr *= (1.0 + (double)x->dispersion * (double)(k*k));
 double s = sin(M_PI * (double)k / (double)N);
 fr *= (1.0 + (double)x->density * s);
 double fs = (double)x->sr;
 if (fr > 0.45*fs) fr = 0.45*fs;
 m->freq_hz = fr;
 
 double d_s = (m->decay_ms<=0?0.0:m->decay_ms*0.001);
 double r = (d_s<=0.0)? 0.0 : exp(-1.0/(d_s*fs));
 double frel = fr / base; if (frel < 0.1) frel = 0.1;
 double expo = 1.0 + 0.8 * sqrt(frel);
 double r_w = pow(clampf(x->r_mul,0.f,1.f), expo);
 r *= r_w;
 if (r>0.999999) r = 0.999999;
 if (r<0.0) r = 0.0;
 m->rcur = r;
 
 double w = 2.0 * M_PI * fr / fs;
 m->a1 = 2.0 * r * cos(w);
 m->a2 = - (r * r);
 
 double posw = fabs(sin(M_PI * (double)k * (double)clampf(x->position,0.f,1.f)));
 if (posw < 0.05) posw = 0.05;
 int P = (x->aniso_P<1?1:x->aniso_P);
 double aniso = 1.0 + (double)x->aniso_gamma * cos(2.0*M_PI * (double)(k-1) / (double)P);
-if (aniso < 0.25) aniso = 0.25; if (aniso > 2.0) aniso = 2.0;
+if (aniso < 0.25)
+    aniso = 0.25;
+else if (aniso > 2.0)
+    aniso = 2.0;
 double bright = pow(fr/base, (double)clampf(x->bright_slope, -2.f, 2.f));
-if (bright < 0.05) bright = 0.05; if (bright > 20.0) bright = 20.0;
+if (bright < 0.05)
+    bright = 0.05;
+else if (bright > 20.0)
+    bright = 20.0;
 
 m->pos_w = (float)posw;
 m->aniso_w = (float)aniso;
 m->amp_w = (float)(posw * aniso * bright);
 m->r_w = (float)r_w;
 
 mode_update_pan(m);
 m->dirty_coeffs = 0;
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
diff --git a/src/juicy_bank_tilde.c b/src/juicy_bank_tilde.c
index 54355069f537e9adfecfed0f4b88a0901d7e859f..c87394266d49e8175f50fed1e52c68b193c9914d 100644
--- a/src/juicy_bank_tilde.c
+++ b/src/juicy_bank_tilde.c
@@ -192,65 +198,68 @@ int n = (int)(w[6]);
 bank_recalc_body(x);
 
 int N = x->N;
 int nact = 0;
 for (int k=1; k<=N; ++k){
 t_mode *m = &x->m[k-1];
 m->active = (m->gain > 0.0005f) || (fabs(m->env) > 1e-6);
 if (m->active) nact++;
 bank_update_coeffs_one(x, k); // refresh each block
 }
 x->n_active = nact;
 x->mix_scale = (nact>0)? (1.0f / sqrtf((float)nact)) : 1.0f;
 
 for (int i=0;i<n;++i){ outL[i]=0; outR[i]=0; }
 
 const double c_amt = x->contact;
 const double c_soft= x->contact_soft;
 for (int k=1; k<=N; ++k){
 t_mode *m = &x->m[k-1];
 if (!m->active) continue;
 double a1=m->a1, a2=m->a2, y1=m->y1, y2=m->y2;
 double env=m->env;
 double fs = (double)x->sr;
 double att = (m->attack_ms<=0? 0.0 : exp(-1.0/( (double)m->attack_ms*0.001 * fs )));
 double dec = (m->decay_ms <=0? 0.0 : exp(-1.0/( (double)m->decay_ms *0.001 * fs )));
-double gl=m->gl, gr=m->gr;
-double amp = (double)m->gain * (double)x->mix_scale * (double)m->amp_w * (double)m->aniso_w;
-
-for (int i=0;i<n;++i){
-double xin = 0.5*((double)inL[i] + (double)inR[i]) * (double)m->pos_w;
-double tgt = (fabs(xin) > 1e-9) ? 1.0 : 0.0;
-if (tgt > env) env = tgt + (env - tgt) * att; else env = tgt + (env - tgt) * dec;
-double y0 = a1*y1 + a2*y2 + xin;
-y2 = y1; y1 = y0;
-double ynl = contact_shaper(y0, c_amt, c_soft);
-double yamp= amp * env * ynl;
-outL[i] = (t_sample)denorm_fix( (double)outL[i] + yamp * gl );
-outR[i] = (t_sample)denorm_fix( (double)outR[i] + yamp * gr );
-}
-m->y1=y1; m->y2=y2; m->env=env;
+    double gl=m->gl, gr=m->gr;
+    double amp = (double)m->gain * (double)x->mix_scale;
+
+    for (int i=0;i<n;++i){
+        double xin = 0.5*((double)inL[i] + (double)inR[i]) * (double)m->amp_w;
+        xin = contact_shaper(xin, c_amt, c_soft);
+        double tgt = (fabs(xin) > 1e-9) ? 1.0 : 0.0;
+        if (tgt > env)
+            env = tgt + (env - tgt) * att;
+        else
+            env = tgt + (env - tgt) * dec;
+        double y0 = a1*y1 + a2*y2 + xin;
+        y2 = y1; y1 = y0;
+        double yamp = amp * env * y0;
+        outL[i] = (t_sample)denorm_fix( (double)outL[i] + yamp * gl );
+        outR[i] = (t_sample)denorm_fix( (double)outR[i] + yamp * gr );
+    }
+    m->y1=y1; m->y2=y2; m->env=env;
 }
 
 return (t_int *)(w+7);
 }
 
 static void juicy_bank_tilde_dsp(t_juicy_bank_tilde *x, t_signal **sp){
 x->sr = sp[0]->s_sr;
 dsp_add(juicy_bank_tilde_perform, 6, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[3]->s_vec, sp[0]->s_n);
 }
 
 // body setters
 static void set_damping (t_juicy_bank_tilde *x, t_floatarg f){ x->damping=f; }
 static void set_brightness(t_juicy_bank_tilde *x, t_floatarg f){ x->brightness=f; }
 static void set_position (t_juicy_bank_tilde *x, t_floatarg f){ x->position=f; }
 static void set_dispersion(t_juicy_bank_tilde *x, t_floatarg f){ x->dispersion=f; }
 static void set_coupling (t_juicy_bank_tilde *x, t_floatarg f){ x->coupling=f; }
 static void set_density (t_juicy_bank_tilde *x, t_floatarg f){ x->density=f; }
 static void set_anisotropy(t_juicy_bank_tilde *x, t_floatarg f){ x->anisotropy=f; }
 static void set_contact (t_juicy_bank_tilde *x, t_floatarg f){ x->contact=f; }
 
 // edit cursor + per-mode setters
 static void set_idx (t_juicy_bank_tilde *x, t_floatarg f){
 int i = (int)f; if (i<1) i=1; if (i>x->N) i=x->N;
 x->edit_idx = i;
 bank_update_coeffs_one(x, x->edit_idx);
diff --git a/src/juicy_bank_tilde.c b/src/juicy_bank_tilde.c
index 54355069f537e9adfecfed0f4b88a0901d7e859f..c87394266d49e8175f50fed1e52c68b193c9914d 100644
--- a/src/juicy_bank_tilde.c
+++ b/src/juicy_bank_tilde.c
@@ -330,51 +339,51 @@ for (int k=1;k<=x->N;++k){
 t_mode *m=&x->m[k-1];
 post(" id=%d act=%d f=%.2fHz r=%.6f ratio=%.3f key=%d gain=%.3f att=%.1fms dec=%.1fms pan=%.2f ampW=%.3f rW=%.3f posW=%.3f anisoW=%.3f",
 k, m->active, m->freq_hz, m->rcur, m->ratio, (int)(m->keytrack!=0), m->gain, m->attack_ms, m->decay_ms, m->pan,
 m->amp_w, m->r_w, m->pos_w, m->aniso_w);
 }
 }
 static void msg_bug_catch(t_juicy_bank_tilde *x){
 post("=== juicy_bank~ bug_catch ===");
 post("fs=%.1f base=%.2f N=%d damp=%.3f bright=%.3f pos=%.3f disp=%.3f dens=%.3f aniso=%.3f P=%d contact=%.3f soft=%.3f couple_df=%.1f",
 (double)x->sr, x->base_hz, x->N, x->damping, x->brightness, x->position, x->dispersion, x->density, x->anisotropy, x->aniso_P, x->contact, x->contact_soft, x->couple_df);
 post("n_active=%d mix_scale=%.3f edit_idx=%d", x->n_active, x->mix_scale, x->edit_idx);
 for (int k=1;k<=x->N;++k){
 t_mode *m=&x->m[k-1];
 post("id=%d act=%d f=%.2fHz r=%.6f ratio=%.3f key=%d gain=%.3f att=%.1fms dec=%.1fms pan=%.2f ampW=%.3f posW=%.3f anisoW=%.3f",
 k, m->active, m->freq_hz, m->rcur, m->ratio, (int)(m->keytrack!=0), m->gain, m->attack_ms, m->decay_ms, m->pan, m->amp_w, m->pos_w, m->aniso_w);
 }
 bank_emit_body_state(x);
 bank_emit_res_selected(x);
 }
 
 // ctor/dtor
 static void *juicy_bank_tilde_new(t_symbol *s, int argc, t_atom *argv){
 (void)s;
 t_juicy_bank_tilde *x = (t_juicy_bank_tilde*)pd_new(juicy_bank_tilde_class);
 x->N=12; x->base_hz=440;
-x->damping=0; x->brightness=0; x->position=0.5f; x->dispersion=0;
+x->damping=0; x->brightness=0; x->position=0.37f; x->dispersion=0;
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
 m->a1=m->a2=m->y1=m->y2=m->env=0.0; m->gl=1.0; m->gr=0.0; m->freq_hz=0.0; m->rcur=0.0;
 m->amp_w=1.f; m->r_w=1.f; m->pos_w=1.f; m->aniso_w=1.f;
