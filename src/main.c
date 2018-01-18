#include "argparse.h"   // command line parser
#include "iio.h"        // image i/o

#include <stdlib.h>
#include <math.h>       // nans (used as boundary value by bicubic interp)
#include <fftw3.h>      // computes dct
#include <omp.h>

// some macros and data types [[[1

//#define DUMP_INFO

// comment for a simpler version without keeping track of pixel variances
//#define VARIANCES

// comment for uniform aggregation
//#define WEIGHTED_AGGREGATION

#define max(a,b) \
	({ __typeof__ (a) _a = (a); \
	   __typeof__ (b) _b = (b); \
	   _a > _b ? _a : _b; })

#define min(a,b) \
	({ __typeof__ (a) _a = (a); \
	   __typeof__ (b) _b = (b); \
	   _a < _b ? _a : _b; })

// read/write image sequence [[[1
static
float * vio_read_video_float_vec(const char * const path, int first, int last, 
		int *w, int *h, int *pd)
{
	// retrieve size from first frame and allocate memory for video
	int frames = last - first + 1;
	int whc;
	float *vid = NULL;
	{
		char frame_name[512];
		sprintf(frame_name, path, first);
		float *im = iio_read_image_float_vec(frame_name, w, h, pd);

		// size of a frame
		whc = *w**h**pd;

		vid = malloc(frames*whc*sizeof(float));
		memcpy(vid, im, whc*sizeof(float));
		if(im) free(im);
	}

	// load video
	for (int f = first + 1; f <= last; ++f)
	{
		int w1, h1, c1;
		char frame_name[512];
		sprintf(frame_name, path, f);
		float *im = iio_read_image_float_vec(frame_name, &w1, &h1, &c1);

		// check size
		if (whc != w1*h1*c1)
		{
			fprintf(stderr, "Size missmatch when reading frame %d\n", f);
			if (im)  free(im);
			if (vid) free(vid);
			return NULL;
		}

		// copy to video buffer
		memcpy(vid + (f - first)*whc, im, whc*sizeof(float));
		if(im) free(im);
	}

	return vid;
}

static
void vio_save_video_float_vec(const char * const path, float * vid, 
		int first, int last, int w, int h, int c)
{
	const int whc = w*h*c;
	for (int f = first; f <= last; ++f)
	{
		char frame_name[512];
		sprintf(frame_name, path, f);
		float * im = vid + (f - first)*whc;
		iio_save_image_float_vec(frame_name, im, w, h, c);
	}
}


// dct handler [[[1

// dct implementation: using fftw or as a matrix product
enum dct_method {FFTW, MATPROD};

// struct containing the workspaces for the dcts
struct dct_threads
{
	// workspaces for DCT transforms in each thread
	fftwf_plan plan_forward [100];
	fftwf_plan plan_backward[100];
	float *dataspace[100];
	float *datafreq [100];

	// size of the signals
	int width;
	int height;
	int frames;
	int nsignals;
	int nthreads;

	// DCT bases for matrix product implementation 
	// TODO
	
	// which implementation to use
	enum dct_method method;
};

// init dct workspaces [[[2
void dct_threads_init(int w, int h, int f, int n, int t, struct dct_threads * dct_t)
{
#ifdef _OPENMP
	t = min(t, omp_get_max_threads());
	if (t > 100)
	{
		fprintf(stderr,"Error: dct_threads is hard-coded"
		               "for a maximum of 100 threads\n");
		exit(1);
	}
#else
	if (t > 1)
	{
		fprintf(stderr,"Error: dct_threads can't handle"
		               "%d threads (no OpenMP)\n", t);
		exit(1);
	}
#endif

	dct_t->width = w;
	dct_t->height = h;
	dct_t->frames = f;
	dct_t->nsignals = n;
	dct_t->nthreads = t;

   unsigned int N = w * h * f * n;

	// define method based on patch size
//	dct_t->method = (width * height * frames < 32) ? MATPROD : FFTW;
	dct_t->method = FFTW;
//	dct_t->method = MATPROD; // FIXME: MATPROD IS NOT WORKING!

//	fprintf(stderr, "init DCT for %d thread - %d x %d x %d ~ %d\n",nthreads, width, height, frames, nsignals);

	switch (dct_t->method)
	{
		case FFTW:

			for (int i = 0; i < dct_t->nthreads; i++)
			{
				dct_t->dataspace[i] = (float*)fftwf_malloc(sizeof(float) * N);
				dct_t->datafreq [i] = (float*)fftwf_malloc(sizeof(float) * N);
		
				int sz[] = {f, h, w};
		
				fftwf_r2r_kind dct[] = {FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10};
				dct_t->plan_forward[i] = fftwf_plan_many_r2r(3, sz, n,
						dct_t->dataspace[i], NULL, 1, w * h * f, 
						dct_t->datafreq [i], NULL, 1, w * h * f,
//						dct_t->dataspace[i], NULL, n, 1, 
//						dct_t->datafreq [i], NULL, n, 1,
						dct, FFTW_ESTIMATE);
		
				fftwf_r2r_kind idct[] = {FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01};
				dct_t->plan_backward[i] = fftwf_plan_many_r2r(3, sz, n,
						dct_t->datafreq [i], NULL, 1, w * h * f,
						dct_t->dataspace[i], NULL, 1, w * h * f,
//						dct_t->datafreq [i], NULL, n, 1,
//						dct_t->dataspace[i], NULL, n, 1,
						idct, FFTW_ESTIMATE);
			}
			break;
			
		case MATPROD:

			fprintf(stderr, "MATPROD DCT HANDLER IS NOT YET IMPLEMENTED\n");
			break;
	}

}

// delete dct workspaces [[[2
void dct_threads_destroy(struct dct_threads * dct_t)
{
	if (dct_t->nsignals)
	{
		for (int i = 0; i < dct_t->nthreads; i++)
		{
			fftwf_free(dct_t->dataspace[i]);
			fftwf_free(dct_t->datafreq [i]);
			fftwf_destroy_plan(dct_t->plan_forward [i]);
			fftwf_destroy_plan(dct_t->plan_backward[i]);
		}
	}
}

// compute forward dcts [[[2
void dct_threads_forward(float * patch, struct dct_threads * dct_t)
{
   int tid = 0;
#ifdef _OPENMP
   tid = omp_get_thread_num();
#endif

	int w   = dct_t->width;
	int h   = dct_t->height;
	int f   = dct_t->frames;
	int n   = dct_t->nsignals;
	int wh  = w * h;
	int whf = wh * f;
	int N   = whf * n;

	if (N == 0)
		fprintf(stderr, "Attempting to use an uninitialized dct_threads struct.\n");

	switch (dct_t->method)
	{
		case MATPROD: // compute dct via separable matrix products

			fprintf(stderr, "MATPROD DCT HANDLER IS NOT YET IMPLEMENTED\n");
			break;

		case FFTW: // compute dct via fftw
		{
			// copy and compute unnormalized dct
			for (int i = 0; i < N; i++) dct_t->dataspace[tid][i] = patch[i];

			fftwf_execute(dct_t->plan_forward[tid]);

			// copy and orthonormalize
			float norm   = 1.0/sqrt(8.0*(float)whf);
			float isqrt2 = 1.f/sqrt(2.0);

			for (int i = 0; i < N; i++) patch[i] = dct_t->datafreq[tid][i] * norm;

			for (int i = 0; i < n; i++)
			{
				for (int t = 0; t < f; t++)
				for (int y = 0; y < h; y++)
					patch[i*whf + t*wh + y*w] *= isqrt2;

				for (int t = 0; t < f; t++)
				for (int x = 0; x < w; x++)
					patch[i*whf + t*wh + x] *= isqrt2;

				for (int y = 0; y < h; y++)
				for (int x = 0; x < w; x++)
					patch[i*whf + y*w + x] *= isqrt2;
			}

			break;
		}
	}
}

// compute inverse dcts [[[2
void dct_threads_inverse(float * patch, struct dct_threads * dct_t)
{
	int tid = 0;
#ifdef _OPENMP
   tid = omp_get_thread_num();
#endif

	int w   = dct_t->width;
	int h   = dct_t->height;
	int f   = dct_t->frames;
	int n   = dct_t->nsignals;
	int wh  = w * h;
	int whf = wh * f;
	int N   = whf * n;

	if (N == 0)
		fprintf(stderr, "Attempting to use a uninitialized dct_threads struct.\n");

	switch (dct_t->method)
	{
		case MATPROD: // compute dct via separable matrix products

			fprintf(stderr, "MATPROD DCT HANDLER IS NOT YET IMPLEMENTED\n");
			break;

		case FFTW: // compute dct via fftw
		{
			// normalize
			float norm  = 1.0/sqrt(8.0*(float)whf);
			float sqrt2 = sqrt(2.0);

			for (int i = 0; i < n; i++)
			{
				for (int t = 0; t < f; t++)
				for (int y = 0; y < h; y++)
					patch[i*whf + t*wh + y*w] *= sqrt2;

				for (int t = 0; t < f; t++)
				for (int x = 0; x < w; x++)
					patch[i*whf + t*wh + x] *= sqrt2;

				for (int y = 0; y < h; y++)
				for (int x = 0; x < w; x++)
					patch[i*whf + y*w + x] *= sqrt2;
			}

			for (int i = 0; i < N; i++) dct_t->datafreq[tid][i] = patch[i] * norm;

			fftwf_execute(dct_t->plan_backward[tid]);

			for (int i=0; i < N; i++) patch[i] = dct_t->dataspace[tid][i];
		}
	}
}


// window functions [[[1

float * window_function(const char * type, int NN)
{
	const float N = (float)NN;
	const float N2 = (N - 1.)/2.;
	const float PI = 3.14159265358979323846;
	float w1[NN];

	if (strcmp(type, "parzen") == 0)
		for (int n = 0; n < NN; ++n)
		{
			float nc = (float)n - N2;
			w1[n] = (fabs(nc) <= N/4.)
			      ? 1. - 24.*nc*nc/N/N*(1. - 2./N*fabs(nc))
					: 2.*pow(1. - 2./N*fabs(nc), 3.);
		}
	else if (strcmp(type, "welch") == 0)
		for (int n = 0; n < NN; ++n)
		{
			const float nc = ((float)n - N2)/N2;
			w1[n] = 1. - nc*nc;
		}
	else if (strcmp(type, "sine") == 0)
		for (int n = 0; n < NN; ++n)
			w1[n] = sin(PI*(float)n/(N-1));
	else if (strcmp(type, "hanning") == 0)
		for (int n = 0; n < NN; ++n)
		{
			w1[n] = sin(PI*(float)n/(N-1));
			w1[n] *= w1[n];
		}
	else if (strcmp(type, "hamming") == 0)
		for (int n = 0; n < NN; ++n)
			w1[n] = 0.54 - 0.46*cos(2*PI*(float)n/(N-1));
	else if (strcmp(type, "blackman") == 0)
		for (int n = 0; n < NN; ++n)
			w1[n] = 0.42 - 0.5*cos(2*PI*(float)n/(N-1)) + 0.08*cos(4*PI*n/(N-1));
	else if (strcmp(type, "gaussian") == 0)
		for (int n = 0; n < NN; ++n)
		{
			const float s = .4; // scale parameter for the Gaussian
			const float x = ((float)n - N2)/N2/s;
			w1[n] = exp(-.5*x*x);
		}
	else // default is the flat window
		for (int n = 0; n < NN; ++n)
			w1[n] = 1.f;

	// 2D separable window
	float * w2 = malloc(NN*NN*sizeof(float));
	for (int i = 0; i < NN; ++i)
	for (int j = 0; j < NN; ++j)
		w2[i*NN + j] = w1[i]*w1[j];

	return w2;
}




// recursive nl-means algorithm [[[1

// struct for storing the parameters of the algorithm
struct vnlmeans_params
{
	int patch_sz_x;      // spatial  patch size
	int patch_sz_t;      // temporal patch size
	int search_sz_x;     // spatial  search window radius
	int search_sz_t;     // temporal search window radius
	float dista_th;      // patch distance threshold
	float dista_lambda;  // weight of current frame in patch distance
	float beta_x;        // noise multiplier in spatial filtering
	float beta_t;        // noise multiplier in kalman filtering
	bool pixelwise;      // toggle pixel-wise nlmeans
};

// set default parameters as a function of sigma
void vnlmeans_default_params(struct vnlmeans_params * p, float sigma)
{
	/* we trained using two different datasets (both grayscale): 
	 * - derfhd: videos of half hd resolution obtained by downsampling
	 *           some hd videos from the derf database
	 * - derfcif: videos of cif resolution also of the derf db
	 *
	 * we found that the optimal parameters differ. in both cases, the relevant
	 * parameters were the patch distance threshold and the b_t coefficient that
	 * controls the amount of temporal averaging.
	 *
	 * with the derfhd videos, the distance threshold is lower (i.e. patches
	 * are required to be at a smallest distance to be considered 'similar',
	 * and the temporal averaging is higher.
	 *
	 * the reason for the lowest distance threshold is that the derfhd videos 
	 * are smoother than the cif ones. in fact, the cif ones are not properly
	 * sampled and have a considerable amount of aliasing. this high frequencies
	 * increase the distance between similar patches. 
	 *
	 * i don't know which might be the reason for the increase in the temporal 
	 * averaging factor. perhaps that (a) the optical flow can be better estimated
	 * (b) there are more homogeneous regions. in the case of (b), even if the oflow
	 * is not correct, increasing the temporal averaging at these homogeneous regions
	 * might lead to a global decrease in psnr */
#define DERFHD_PARAMS
#ifdef DERFHD_PARAMS
	if (p->patch_sz_x   < 0) p->patch_sz_x   = 8;  // not tuned
	if (p->patch_sz_t   < 0) p->patch_sz_t   = 1;  // not tuned
	if (p->search_sz_x  < 0) p->search_sz_x  = 10; // not tuned
	if (p->search_sz_t  < 0) p->search_sz_t  = 1;  // not tuned
	if (p->dista_th     < 0) p->dista_th     = .5*sigma + 15.0;
	if (p->dista_lambda < 0) p->dista_lambda = 1.0;
	if (p->beta_x       < 0) p->beta_x       = 3.0;
	if (p->beta_t       < 0) p->beta_t       = 0.05*sigma + 6.0;
#else // DERFCIF_PARAMS
	if (p->patch_sz_x   < 0) p->patch_sz_x   = 8;  // not tuned
	if (p->patch_sz_t   < 0) p->patch_sz_t   = 1;  // not tuned
	if (p->search_sz_x  < 0) p->search_sz_x  = 10; // not tuned
	if (p->search_sz_t  < 0) p->search_sz_t  = 1;  // not tuned
	if (p->dista_th     < 0) p->dista_th     = (60. - 38.)*(sigma - 10.) + 38.0;
	if (p->dista_lambda < 0) p->dista_lambda = 1.0;
	if (p->beta_x       < 0) p->beta_x       = 2.4;
	if (p->beta_t       < 0) p->beta_t       = 4.5;
#endif
}

// denoise frame t
void vnlmeans_frame(float *deno1, float *nisy1, float *flow1,
		int w, int h, int ch, float sigma,
		const struct vnlmeans_params prms, int frame)
{
	// definitions [[[2

	const int psz = prms.patch_sz_x;
	const int step = prms.pixelwise ? 1 : psz/2;
//	const int step = prms.pixelwise ? 1 : psz;
	const float sigma2 = sigma * sigma;
	const float dista_th2 = prms.dista_th * prms.dista_th;
	const float beta_x = prms.beta_x;
	const int psz_t = min(frame + 1, prms.patch_sz_t);
	const int wsz_t = min(frame + 2 - psz_t, prms.search_sz_t);

	/* deno1: frames [frame - psz_t + 1, ..., frame]; output buffer
	 * nisy1: frames [0, ..., frame]; only last frame is noisy
	 * flow1: backwards optical flow from [0, ..., frame]
	 */

	// aggregation weights (not necessary for pixel-wise nlmeans)
	float *aggr1 = prms.pixelwise ? NULL : malloc(w*h*sizeof(float));

	// set output and aggregation weights to 0
	if (aggr1) for (int i = 0; i < w*h; ++i) aggr1[i] = 0.;
	for (int t = 0; t < psz_t; ++t)
	for (int i = 0; i < w*h*ch; ++i)
		deno1[i - t*w*h*ch] = 0.;

	// compute a window (to reduce blocking artifacts)
	float *window = window_function("gaussian", psz);
	float W[psz][psz];
	for (int i = 0; i < psz; ++i)
	for (int j = 0; j < psz; ++j)
		W[i][j] = window[i*psz + j];
	free(window);

	// noisy and clean patches at point p (as VLAs in the stack!)
	float P[ch][psz_t][psz][psz]; // noisy patch at position p in frame t

	// wrap images with nice pointers to vlas
	float (*a1)[w]        = (void *)aggr1;       // aggregation weights at t
	float (*d1)[h][w][ch] = (void *)deno1;       // denoised frame t (output)
	float (*of)[h][w][2]  = (void *)flow1;       // denoised frame t (output)
	const float (*n1)[h][w][ch] = (void *)nisy1; // noisy frame at t

	// initialize dct workspaces (we will compute the dct of two patches)
	float Q[ch][psz_t][psz][psz]; // noisy patch at q and clean patch at t-1
	struct dct_threads dcts[1];
#ifdef _OPENMP
	const int nthreads = omp_get_max_threads();
#else
	const int nthreads = 1;
#endif
	dct_threads_init(psz, psz, psz_t, ch, nthreads, dcts); // 3D DCT

	// statistics
	float M1 [ch][psz_t][psz][psz]; // average patch at t
	float V1 [ch][psz_t][psz][psz]; // variance at t

#ifdef DUMP_INFO
	float *np0image = (float *)malloc(w*h*sizeof(float));
	float (*np0im)[w] = (void *)np0image;
	for (int i = 0; i < w*h; i++) np0image[i] = 0.f;
#endif

	// loop on image patches [[[2
	const int parallel_step = step * (psz/step);
	for (int oy = 0; oy < psz; oy += step) // split in grids of non-overlapping
	for (int ox = 0; ox < psz; ox += step) // patches (for parallelization)
	#pragma omp parallel for private(Q,P,M1,V1)
	for (int py = oy; py < h - psz + 1; py += parallel_step) // FIXME: boundaries
	for (int px = ox; px < w - psz + 1; px += parallel_step) // may be skipped
	{
		//	load target patch [[[3
		for (int ht = 0; ht < psz_t; ++ht)
		for (int hy = 0; hy < psz; ++hy)
		for (int hx = 0; hx < psz; ++hx)
		for (int c  = 0; c  < ch ; ++c )
		{
			P [c][ht][hy][hx] = n1[-ht][py + hy][px + hx][c];
			M1[c][ht][hy][hx] = 0.;
			V1[c][ht][hy][hx] = 0.;
		}

		// gather spatio-temporal statistics: loop on search region [[[3
		int np1 = 0; // number of similar patches with no valid previous patch
		if (dista_th2)
		{
			const int wsz = prms.search_sz_x;
			const int wt[2] = {0, wsz_t};
			float cx = px, cy = py;
			int icx = px, icy = py;
			for (int qt = wt[0]; qt < wt[1]; ++qt)
			{
				// compute patch trajectory using optical flow
				if (of && qt)
				{
					cx += of[-qt+1][icy][icx][0];
					cy += of[-qt+1][icy][icx][1];
					icx = max(0, min(w, round(cx)));
					icy = max(0, min(h, round(cy)));
				}

				// spatial region centered at patch trajectory
				const int wx[2] = {max(icx - wsz, 0), min(icx + wsz, w - psz) + 1};
				const int wy[2] = {max(icy - wsz, 0), min(icy + wsz, h - psz) + 1};
				for (int qy = wy[0]; qy < wy[1]; ++qy)
				for (int qx = wx[0]; qx < wx[1]; ++qx)
				{
					// compute patch distance [[[4
					float ww = 0; // patch distance is saved here
					for (int ht = 0; ht < psz_t; ++ht)
					for (int hy = 0; hy < psz; ++hy)
					for (int hx = 0; hx < psz; ++hx)
					for (int c  = 0; c  < ch ; ++c )
					{
						// store patch at q
						Q[c][ht][hy][hx] = n1[-qt - ht][qy + hy][qx + hx][c];

						// compute distance, compensating for noise
						const float e1 = Q[c][ht][hy][hx] - P[c][ht][hy][hx];
						ww += e1 * e1 - sigma2 * ((ht == 0) ? (qt == 0) ? 2 : 1 : 0);
					}
	
					// normalize distance by number of pixels in patch
					ww = max(ww / ((float)psz_t*psz*psz*ch), 0);
	
					// if patch at q is similar to patch at p, update statistics [[[4
					if (ww <= dista_th2)
					{
						np1++;
	
						// compute dct (output in Q)
						dct_threads_forward((float *)Q, dcts);
	
						// compute means and variances.
						// to compute the variances in a single pass over the search
						// region we use Welford's method.
						const float inp1 = 1./(float)np1;
						for (int c  = 0; c  < ch ; ++c )
						for (int ht = 0; ht < psz_t; ++ht)
						for (int hy = 0; hy < psz; ++hy)
						for (int hx = 0; hx < psz; ++hx)
						{
							const float p = Q[c][ht][hy][hx];
							const float oldM1 = M1[c][ht][hy][hx];
							const float delta = p - oldM1;
	
							M1[c][ht][hy][hx] += delta * inp1;
							V1[c][ht][hy][hx] += delta * (p - M1[c][ht][hy][hx]); 
							V1[c][ht][hy][hx] -= sigma2 * ((ht == 0 && qt == 0) ? 1 : 0);
						}
					} // ]]]4
				}
			}

			// correct variance [[[4
			const float inp1 = 1./(float)np1;
			for (int c  = 0; c  < ch ; ++c )
			for (int ht = 0; ht < psz_t; ++ht)
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
				V1[c][ht][hy][hx] *= inp1;

			// ]]]4
		}
		else // dista_th2 == 0
		{
			// local version: single point estimate of variances [[[4
			//                the mean M1 is assumed to be 0

			for (int c  = 0; c  < ch ; ++c )
			for (int ht = 0; ht < psz_t; ++ht)
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
				Q[c][ht][hy][hx] = P[c][ht][hy][hx];

			// compute dct (output in N1D0)
			dct_threads_forward((float *)Q, dcts);

			// patch statistics (point estimate)
			for (int c  = 0; c  < ch ; ++c )
			for (int ht = 0; ht < psz_t; ++ht)
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
			{
				float p = Q[c][ht][hy][hx];
				V1[c][ht][hy][hx] = p * p - sigma2 * (ht == 0 ? 1 : 0);
			}//]]]4
		}

		// filter current patch [[[3

		// compute dct (computed in place in N1D0)
		dct_threads_forward((float *)P, dcts);

		// spatial nl-dct using statistics in M1 V1
		float vp = 0;
		for (int c  = 0; c  < ch ; ++c )
		for (int ht = 0; ht < psz_t; ++ht)
		for (int hy = 0; hy < psz; ++hy)
		for (int hx = 0; hx < psz; ++hx)
		{
			// prediction variance (sigma2 was subsracted bfefrom transition variance)
			float v = max(0.f, V1[c][ht][hy][hx]);

			// wiener filter
			float a = v / (v + beta_x * sigma2);
			if (a < 0) printf("a = %f v = %f ", a, v);
			if (a > 1) printf("a = %f v = %f ", a, v);

			// variance of filtered patch
			vp += a * a * v;

			/* thresholding instead of empirical Wiener filtering
			float a = (hy != 0 || hx != 0) ?
			//	(N1D0[c][hy][hx] * N1D0[c][hy][hx] > 3 * sigma2) : 1;
				(v > 1 * sigma2) : 1;
			float a = (hy != 0 || hx != 0) ?
			vp += a;*/

			// filter
			P[c][ht][hy][hx] = a*P[c][ht][hy][hx] + (1 - a)*M1[c][ht][hy][hx];
		}

		// invert dct (output in N1D0)
		dct_threads_inverse((float *)P, dcts);

		// aggregate denoised patch on output image [[[3
		if (a1)
		{
#ifdef WEIGHTED_AGGREGATION
			const float w = 1.f/vp;
#else
			const float w = 1.f;
#endif
			// patch-wise denoising: aggregate the whole denoised patch
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
			{
				a1[py + hy][px + hx] += w * W[hy][hx];
				for (int ht = 0; ht < psz_t; ++ht)
				for (int c = 0; c < ch ; ++c )
					d1[-ht][py + hy][px + hx][c] += w * W[hy][hx] * P[c][ht][hy][hx];
			}
		}
		else 
			// pixel-wise denoising: aggregate only the central pixel
			for (int c = 0; c < ch ; ++c )
				d1[0][py + psz/2][px + psz/2][c] += P[c][0][psz/2][psz/2];

		// ]]]3
	}

	// normalize output [[[2
	if (aggr1)
	for (int t = 0; t < psz_t; ++t) 
	for (int i = 0, j = 0; i < w*h; ++i) 
	for (int c = 0; c < ch ; ++c, ++j) 
		deno1[j - t*w*h*ch] /= aggr1[i];

#ifdef DUMP_INFO
	{
		char name[512];
		sprintf(name, "occ.%03d.tif", frame);
		iio_save_image_float_vec(name, np0image, w, h, 1);
	}
#endif

	// free allocated mem and quit
	dct_threads_destroy(dcts);
	if (aggr1) free(aggr1);

	return; // ]]]2
}

// main funcion [[[1

// 'usage' message in the command line
static const char *const usages[] = {
	"vnlmeans [options] [[--] args]",
	"vnlmeans [options]",
	NULL,
};

int main(int argc, const char *argv[])
{
	// parse command line [[[2

	// command line parameters and their defaults
	const char *nisy_path = NULL;
	const char *deno_path = NULL;
	const char *flow_path = NULL;
	const char *occl_path = NULL;
	int fframe = 0, lframe = -1;
	float sigma = 0.f;
	bool verbose = false;
	struct vnlmeans_params prms;
	prms.patch_sz_x   = -1; // -1 means automatic value
	prms.patch_sz_t   = -1;
	prms.search_sz_x  = -1;
	prms.search_sz_t  = -1;
	prms.dista_th     = -1.;
	prms.beta_x       = -1.;
	prms.beta_t       = -1.;
	prms.dista_lambda = -1.;
	prms.pixelwise = false;

	// configure command line parser
	struct argparse_option options[] = {
		OPT_HELP(),
		OPT_GROUP("Algorithm options"),
		OPT_STRING ('i', "nisy"    , &nisy_path, "noisy input path (printf format)"),
		OPT_STRING ('o', "flow"    , &flow_path, "backward flow path (printf format)"),
		OPT_STRING ('k', "occl"    , &occl_path, "flow occlusions mask (printf format)"),
		OPT_STRING ('d', "deno"    , &deno_path, "denoised output path (printf format)"),
		OPT_INTEGER('f', "first"   , &fframe, "first frame"),
		OPT_INTEGER('l', "last"    , &lframe , "last frame"),
		OPT_FLOAT  ('s', "sigma"   , &sigma, "noise standard dev"),
		OPT_INTEGER('p', "patch_x" , &prms.patch_sz_x, "spatial  patch size"),
		OPT_INTEGER( 0 , "patch_t" , &prms.patch_sz_t, "temporal patch size"),
		OPT_INTEGER('w', "search_x", &prms.search_sz_x, "spatial  search region radius"),
		OPT_INTEGER( 0 , "search_t", &prms.search_sz_t, "temporal search region radius"),
		OPT_FLOAT  ( 0 , "dth"     , &prms.dista_th, "patch distance threshold"),
		OPT_FLOAT  ( 0 , "beta_x"  , &prms.beta_x, "noise multiplier in spatial filtering"),
		OPT_FLOAT  ( 0 , "beta_t"  , &prms.beta_t, "noise multiplier in kalman filtering"),
		OPT_FLOAT  ( 0 , "lambda"  , &prms.dista_lambda, "noisy patch weight in patch distance"),
		OPT_BOOLEAN( 0 , "pixel"   , &prms.pixelwise, "toggle pixel-wise denoising"),
		OPT_GROUP("Program options"),
		OPT_BOOLEAN('v', "verbose", &verbose, "verbose output"),
		OPT_END(),
	};

	// parse command line
	struct argparse argparse;
	argparse_init(&argparse, options, usages, 0);
	argparse_describe(&argparse, "\nA video denoiser based on non-local means.", "");
	argc = argparse_parse(&argparse, argc, argv);

	// default value for noise-dependent params
	vnlmeans_default_params(&prms, sigma);

	// print parameters
	if (verbose)
	{
		printf("parameters:\n");
		printf("\tnoise  %f\n", sigma);
		printf("\t%s-wise mode\n", prms.pixelwise ? "pixel" : "patch");
		printf("\tpatch_x   %d\n", prms.patch_sz_x);
		printf("\tpatch_t   %d\n", prms.patch_sz_t);
		printf("\tsearch_x  %d\n", prms.search_sz_x);
		printf("\tsearch_t  %d\n", prms.search_sz_t);
		printf("\tdth       %g\n", prms.dista_th);
		printf("\tlambda    %g\n", prms.dista_lambda);
		printf("\tbeta_x    %g\n", prms.beta_x);
		printf("\tbeta_t    %g\n", prms.beta_t);
		printf("\n");
#ifdef VARIANCES
		printf("\tVARIANCES ON\n");
#endif
#ifdef WEIGHTED_AGGREGATION
		printf("\tWEIGHTED_AGGREGATION ON\n");
#endif
	}

	// load data [[[2
	if (verbose) printf("loading video %s\n", nisy_path);
	int w, h, c; //, frames = lframe - fframe + 1;
	float * nisy = vio_read_video_float_vec(nisy_path, fframe, lframe, &w, &h, &c);
	{
		if (!nisy)
			return EXIT_FAILURE;
	}

	// load optical flow
	float * flow = NULL;
	if (flow_path)
	{
		if (verbose) printf("loading flow %s\n", flow_path);
		int w1, h1, c1;
		flow = vio_read_video_float_vec(flow_path, fframe, lframe, &w1, &h1, &c1);

		if (!flow)
		{
			if (nisy) free(nisy);
			return EXIT_FAILURE;
		}

		if (w*h != w1*h1 || c1 != 2)
		{
			fprintf(stderr, "Video and optical flow size missmatch\n");
			if (nisy) free(nisy);
			if (flow) free(flow);
			return EXIT_FAILURE;
		}
	}

	// load occlusion masks
	float * occl = NULL;
	if (flow_path && occl_path)
	{
		if (verbose) printf("loading occl. mask %s\n", occl_path);
		int w1, h1, c1;
		occl = vio_read_video_float_vec(occl_path, fframe, lframe, &w1, &h1, &c1);

		if (!occl)
		{
			if (nisy) free(nisy);
			if (flow) free(flow);
			return EXIT_FAILURE;
		}

		if (w*h != w1*h1 || c1 != 1)
		{
			fprintf(stderr, "Video and occl. masks size missmatch\n");
			if (nisy) free(nisy);
			if (flow) free(flow);
			if (occl) free(occl);
			return EXIT_FAILURE;
		}
	}

	// run denoiser [[[2
	const int whc = w*h*c, wh2 = w*h*2;
	const int patch_t  = prms.patch_sz_t;

	float * deno = malloc(whc * patch_t * sizeof(float));
	for (int f = fframe; f <= lframe; ++f)
	{
		if (verbose) printf("processing frame %d\n", f);

		// run denoising
		float *nisy1 = nisy + (f - fframe)*whc;
		float *flow1 = flow + (f - fframe)*wh2;
		float *deno1 = deno + (patch_t - 1)*whc; // point to last frame of deno1

		vnlmeans_frame(deno1, nisy1, flow1, w, h, c, sigma, prms, f-fframe);

		// copy denoised frame f to video
		memcpy(nisy1, deno1, whc*sizeof(float));
	}

	// save output [[[2
	vio_save_video_float_vec(deno_path, nisy, fframe, lframe, w, h, c);

	if (deno) free(deno);
	if (nisy) free(nisy);
	if (flow) free(flow);
	if (occl) free(occl);

	return EXIT_SUCCESS; // ]]]2
}

// vim:set foldmethod=marker:
// vim:set foldmarker=[[[,]]]:
