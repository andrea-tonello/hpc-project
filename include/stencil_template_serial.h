/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>  // String operations: memset, strcpy, etc.
#include <unistd.h>  // POSIX API: getopt for command-line parsing
#include <getopt.h>  // GNU extension for advanced option parsing
#include <time.h>
#include <float.h>   // Floating-point limits: DBL_MAX, DBL_MIN
#include <math.h>



#define NORTH 0
#define SOUTH 1
#define EAST  2
#define WEST  3

#define SEND 0
#define RECV 1

#define OLD 0
#define NEW 1

#define _x_ 0
#define _y_ 1

typedef unsigned int uint;

// ============================================================
//
// function prototypes

int initialize (
    int,
    char   **,
	int     *,
	int     *,
	int     *,
	int     *,
	int    **,
	double  *,
	double **,
    int     *,
    int     *
);

// Takes two pointers: one to `double` array (the grid data), one to `int` array (heat source positions).
int memory_release ( double *, int * );


extern int inject_energy ( 
    const int,
    const int,
	const int *,
	const double,
	const int [2],
    double * 
);

extern int update_plane ( 
    const int,
	const int [2],
    const double *,
	double * 
);

extern int get_total_energy( 
    const int [2],
    const double *,
    double * 
);


// ============================================================
//
// function definition for inline functions

// `inline`: suggests to the compiler "copy-paste this function's code at each call site 
// instead of actually calling it" for performance. Small functions benefit from inlining.

// One call represents one injection event. How often it's called depends on injection_frequency
inline int inject_energy ( 
    const int periodic,     // Boolean (0 or 1) - are boundaries periodic?
    const int Nsources,     // How many heat sources
    const int *Sources,     // Pointer to array of source coordinates `[x0, y0, x1, y1, ...]`
    const double energy,    // How much heat to inject at each source
	const int mysize[2],    // Grid dimensions [width, height]
    double *plane           // Pointer to the 2D grid data (stored as 1D array)
)
{
   #define IDX( i, j ) ( (j)*(mysize[_x_]+2) + (i) )

    for (int s = 0; s < Nsources; s++) {
        
        // Sources are stored as [x0, y0, x1, y1, x2, y2, ...]
        // For source s: x is at Sources[2*s], y is at Sources[2*s+1]
        int x = Sources[2*s];       // x-coordinate of source s
        int y = Sources[2*s+1];     // y-coordinate of source s
        plane[IDX(x, y)] += energy;

        // if periodic: if a source is at the edge, we also inject energy at the opposite edge
        if ( periodic )
            {
                if ( x == 1 )
                    plane[IDX(mysize[_x_]+1, y)] += energy;
                if ( x == mysize[_x_] )
                    plane[IDX(0, y)] += energy;
                if ( y == 1 )
                    plane[IDX(x, mysize[_y_]+1)] += energy;
                if ( y == mysize[_y_] )
                    plane[IDX(x, 0)] += energy;
            }
    }
   // Remove the macro definition so it doesn't leak out and conflict with other code
   #undef IDX
    
    return 0;
}



inline int update_plane ( 
    const int periodic, 
    const int size[2],
    const double *old,  // Pointer to current timestep data (read-only due to `const`)
    double *new         // Pointer to updated timestep data (will be written)
)
/*
 * Calculate the new energy values.
 * The old plane contains the current data, 
 * the new plane will store the updated data
 *
 * NOTE: in parallel, every MPI task will perform the calculation for its patch
 */
{
    // `register`: Old keyword (mostly ignored by modern compilers) suggesting 
    // "keep this variable in CPU register for speed" rather than memory.
    const int register fxsize = size[_x_]+2;  // full x size (with ghosts)
    const int register fysize = size[_y_]+2;  // full y size
    const int register xsize = size[_x_];     // interior x size
    const int register ysize = size[_y_];     // interior y size
    
   #define IDX( i, j ) ( (j)*fxsize + (i) )

    // HINT: you may attempt to
    //       (i)  manually unroll the loop
    //       (ii) ask the compiler to do it
    // for instance
    // #pragma GCC unroll 4
    //
    // HINT: in any case, the following loop is a good candidate for omp parallelization

    // Loop: `i` and `j` go from `1` to `size`, which are the interior points only. 
    // We don't update ghost cells here (they get filled from neighbors or periodic copies).
    for (int j = 1; j <= ysize; j++)
        for ( int i = 1; i <= xsize; i++)
            {
                //
                // five-points stencil formula
                //

                
                // Simpler stencil with no explicit diffusivity
                // Always conserve the smoothed quantity
                          
                // alpha mimics how easy the heat travels: "keep 60% of the point's current value"
                double alpha = 0.6;
                double result = old[ IDX(i,j) ] *alpha;

                // Heat diffusion:
                // - Take the average of left+right neighbors: `(old[i-1,j] + old[i+1,j])/2`
                // - Divide by 4 to split diffusion between x and y directions: `/4.0`
                // - Multiply by `(1-alpha) = 0.4` - this is the 40% that diffuses from neighbors
                // - Same for top+bottom neighbors in y-direction
                // - Add both contributions
                double sum_i  = (old[IDX(i-1, j)] + old[IDX(i+1, j)]) / 4.0 * (1-alpha);
                double sum_j  = (old[IDX(i, j-1)] + old[IDX(i, j+1)]) / 4.0 * (1-alpha);
                result += (sum_i + sum_j);
                

                /*

                  // implentation from the derivation of
                  // 3-points 2nd order derivatives
                  // however, that should depends on an adaptive
                  // time-stepping so that given a diffusivity
                  // coefficient the amount of energy diffused is "small"
                  // however the imlic methods are not stable
                  
               #define alpha_guess 0.5     // mimic the heat diffusivity

                double alpha = alpha_guess;
                double sum = old[IDX(i,j)];
                
                int   done = 0;
                do
                    {                
                        double sum_i = alpha * (old[IDX(i-1, j)] + old[IDX(i+1, j)] - 2*sum);
                        double sum_j = alpha * (old[IDX(i, j-1)] + old[IDX(i, j+1)] - 2*sum);
                        result = sum + ( sum_i + sum_j);
                        double ratio = fabs((result-sum)/(sum!=0? sum : 1.0));
                        done = ( (ratio < 2.0) && (result >= 0) );    // not too fast diffusion and
                                                                     // not so fast that the (i,j)
                                                                     // goes below zero energy
                        alpha /= 2;
                    }
                while ( !done );
                */

                new[ IDX(i,j) ] = result;
                
            }

    if ( periodic )
        /*
         * propagate boundaries if they are periodic
         *
         * NOTE: when is that needed in distributed memory, if any?
         */
        {
            for ( int i = 1; i <= xsize; i++ )
                {
                    new[ i ] = new[ IDX(i, ysize) ];
                    new[ IDX(i, ysize+1) ] = new[ IDX(i, 1) ];
                }
            for ( int j = 1; j <= ysize; j++ )
                {
                    new[ IDX( 0, j) ] = new[ IDX(xsize, j) ];
                    new[ IDX( xsize+1, j) ] = new[ IDX(1, j) ];
                }
        }
    
    return 0;

   #undef IDX
}

 

inline int get_total_energy ( 
    const int size[2],
    const double *plane,
    double *energy 
)
/*
 * NOTE: this routine a good candidate for openmp parallelization
 */
{
    const int register xsize = size[_x_];
    
   #define IDX( i, j ) ( (j)*(xsize+2) + (i) )

   // Conditional compilation: If `LONG_ACCURACY` is defined (via `-DLONG_ACCURACY` compiler flag), 
   // use `long double` (128-bit) for better precision. Otherwise use `double` (64-bit).
   #if defined(LONG_ACCURACY)    
    long double totenergy = 0;
   #else
    double totenergy = 0;    
   #endif

    // HINT: you may attempt to
    //       (i)  manually unroll the loop
    //       (ii) ask the compiler to do it
    // for instance
    // #pragma GCC unroll 4
    for ( int j = 1; j <= size[_y_]; j++ )
        for ( int i = 1; i <= size[_x_]; i++ )
            totenergy += plane[ IDX(i, j) ];
    
   #undef IDX

    *energy = (double)totenergy;
    return 0;
}
                            
