/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <math.h>

#include <omp.h>
#include <mpi.h>


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

// Syntax means: writing `vec2_t myarr` is now equivalent to
// writing `uint myarr[2]`, i.e. "array of 2 unsigned ints"
typedef uint vec2_t[2];

// `buffers_t` is an alias for "array of 4 pointers to double". Used for communication buffers
typedef double *restrict buffers_t[4];

// `plane_t` bundles:
// - `data`: Pointer to the grid values (the actual numbers)
// - `size`: The dimensions of the grid [width, height]
typedef struct {
    double * restrict data;
    vec2_t size;
} plane_t;


extern int inject_energy ( 
    const int,
    const int,
	const vec2_t *,  // was: const int *
    const double,
          plane_t *, // was: double *
    const vec2_t     // NEW: the MPI grid
);


extern int update_plane_interior (
    const int,
    const vec2_t,    // MPI grid
    const plane_t *,
          plane_t *
);

extern int update_plane_border (
    const int,
    const vec2_t,    // MPI grid
    const plane_t *,
          plane_t *
);


extern int get_total_energy( plane_t *, double * );

int initialize (
    MPI_Comm *,    // NEW: MPI communicator
    int       ,    // NEW: Rank (my process ID)
    int       ,    // NEW: Ntasks (total processes)
    int       ,
    char    **,
    vec2_t   *,
    vec2_t   *,    // NEW: N (MPI grid size)
    int      *,    // periodic
    int      *,    // output_energy_stat
    int      *,    // overlap (comm/comp)
    int      *,
    int      *,
    int      *,
    int      *,
    vec2_t  **,
    double   *,
    plane_t  *,
    buffers_t *    // NEW: communication buffers
); 


int memory_release ( buffers_t *, plane_t * );


// `output_energy_stat`: NEW function - uses `MPI_Reduce` to gather energy from all processes.
// vs serial: serial just called `printf` directly. Parallel needs to sum up energies from all processes first.
int output_energy_stat ( 
    int      ,
    plane_t *,
    double   ,
    int      ,
    MPI_Comm *
);


inline int inject_energy ( 
    const int      periodic,
    const int      Nsources,
    const vec2_t  *Sources,
    const double   energy,
          plane_t *plane,
    const vec2_t   N          // NEW `N[_x_]` is the number of MPI tasks in the x-direction
)
{
    const uint register sizex = plane->size[_x_]+2;
    const uint register psx = plane->size[_x_];
    const uint register psy = plane->size[_y_];
    double * restrict data = plane->data;

   #define IDX( i, j ) ( (j)*sizex + (i) )

    // Hoist loop-invariant periodic checks outside the source loop
    const int periodic_x = periodic && (N[_x_] == 1);
    const int periodic_y = periodic && (N[_y_] == 1);

    for (int s = 0; s < Nsources; s++)
        {
            int x = Sources[s][_x_];
            int y = Sources[s][_y_];

            data[ IDX(x,y) ] += energy;

            if ( periodic_x )
            {
                if ( x == 1 )
                    data[IDX(psx+1, y)] += energy;
                if ( x == psx )
                    data[IDX(0, y)] += energy;
            }
            if ( periodic_y )
            {
                if ( y == 1 )
                    data[IDX(x, psy+1)] += energy;
                if ( y == psy )
                    data[IDX(x, 0)] += energy;
            }
        }
 #undef IDX
    
  return 0;
}



// update_plane_interior: compute cells that do NOT depend on ghost cells.
// These are cells where 2 <= i <= xsize-1 AND 2 <= j <= ysize-1.
// If xsize <= 2 or ysize <= 2, the loops are empty and border handles everything.
inline int update_plane_interior (
    const int      periodic,
    const vec2_t   N,
    const plane_t *oldplane,
          plane_t *newplane
)
{
    uint register fxsize = oldplane->size[_x_]+2;
    uint register xsize = oldplane->size[_x_];
    uint register ysize = oldplane->size[_y_];

   #define IDX( i, j ) ( (j)*fxsize + (i) )

    double * restrict old = oldplane->data;
    double * restrict new = newplane->data;

    const double alpha_self  = 0.5;    // weight for center cell
    const double alpha_neigh = 0.125;  // weight for each neighbor (1/4 * 1/2)

    // 2D tiling: TILE chosen so old-plane tile data (TILE+2)^2 * 8 fits in L1 (48 KB).
    // TILE=64 -> (66)^2 * 8 = 34 KB < 48 KB. All stencil reads are L1 hits.
   #define TILE 64

    #pragma omp parallel for collapse(2) schedule(static)
    for (uint jj = 2; jj < ysize; jj += TILE)
        for (uint ii = 2; ii < xsize; ii += TILE)
        {
            uint j_end = (jj + TILE < ysize) ? jj + TILE : ysize;
            uint i_end = (ii + TILE < xsize) ? ii + TILE : xsize;

            for (uint j = jj; j < j_end; j++)
            {
                #pragma omp simd
                for (uint i = ii; i < i_end; i++)
                {
                    new[ IDX(i,j) ] =
                        old[ IDX(i,j) ] * alpha_self + ( old[IDX(i-1, j)] + old[IDX(i+1, j)] +
                                                         old[IDX(i, j-1)] + old[IDX(i, j+1)] ) * alpha_neigh;
                }
            }
        }

   #undef TILE

   #undef IDX
    return 0;
}


// update_plane_border: compute only the border cells (i==1, i==xsize, j==1, j==ysize).
// These cells depend on ghost data, so call this AFTER MPI_Waitall + unpack.
// Also handles periodic boundary copies when a dimension has only 1 MPI task.
inline int update_plane_border (
    const int      periodic,
    const vec2_t   N,
    const plane_t *oldplane,
          plane_t *newplane
)
{
    uint register fxsize = oldplane->size[_x_]+2;

    uint register xsize = oldplane->size[_x_];
    uint register ysize = oldplane->size[_y_];

   #define IDX( i, j ) ( (j)*fxsize + (i) )

    double * restrict old = oldplane->data;
    double * restrict new = newplane->data;

    const double alpha_self  = 0.5;    // weight for center cell
    const double alpha_neigh = 0.125;  // weight for each neighbor (1/4 * 1/2)

    // Top row (j=1, all columns)
    #pragma omp parallel for schedule(static)
    for (uint i = 1; i <= xsize; i++)
        new[ IDX(i,1) ] =
            old[ IDX(i,1) ] * alpha_self + ( old[IDX(i-1, 1)] + old[IDX(i+1, 1)] +
                                              old[IDX(i, 0)]   + old[IDX(i, 2)] ) * alpha_neigh;

    // Bottom row (j=ysize, all columns)
    if (ysize > 1)
        #pragma omp parallel for schedule(static)
        for (uint i = 1; i <= xsize; i++)
            new[ IDX(i,ysize) ] =
                old[ IDX(i,ysize) ] * alpha_self + ( old[IDX(i-1, ysize)] + old[IDX(i+1, ysize)] +
                                                      old[IDX(i, ysize-1)] + old[IDX(i, ysize+1)] ) * alpha_neigh;

    // Left column (i=1, skip corners already done by top/bottom rows)
    #pragma omp parallel for schedule(static)
    for (uint j = 2; j <= ysize - 1; j++)
        new[ IDX(1,j) ] =
            old[ IDX(1,j) ] * alpha_self + ( old[IDX(0, j)]   + old[IDX(2, j)] +
                                              old[IDX(1, j-1)] + old[IDX(1, j+1)] ) * alpha_neigh;

    // Right column (i=xsize, skip corners already done by top/bottom rows)
    if (xsize > 1)
        #pragma omp parallel for schedule(static)
        for (uint j = 2; j <= ysize - 1; j++)
            new[ IDX(xsize,j) ] =
                old[ IDX(xsize,j) ] * alpha_self + ( old[IDX(xsize-1, j)] + old[IDX(xsize+1, j)] +
                                                      old[IDX(xsize, j-1)] + old[IDX(xsize, j+1)] ) * alpha_neigh;

    // Periodic boundary copies (only when a single MPI task spans that dimension)
    if ( periodic )
        {
            if ( N[_x_] == 1 )
            {
                for ( int j = 1; j <= ysize; j++ )
                    {
                        new[ IDX(0, j) ] = new[ IDX(xsize, j) ];
                        new[ IDX(xsize+1, j) ] = new[ IDX(1, j) ];
                    }
            }

            if ( N[_y_] == 1 )
            {
                for ( int i = 1; i <= xsize; i++ )
                    {
                        new[ i ] = new[ IDX(i, ysize) ];
                        new[ IDX(i, ysize+1) ] = new[ IDX(i, 1) ];
                    }
            }
        }

   #undef IDX
    return 0;
}



inline int get_total_energy( 
    plane_t *plane,                         
    double  *energy 
)
/*
 * NOTE: this routine a good candiadate for openmp
 *       parallelization
 */
{
    const int register xsize = plane->size[_x_];
    const int register ysize = plane->size[_y_];
    const int register fsize = xsize+2;

    double * restrict data = plane->data;
    
   #define IDX( i, j ) ( (j)*fsize + (i) )

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
    #pragma omp parallel for collapse(2) schedule(static) reduction(+:totenergy)
    for ( int j = 1; j <= ysize; j++ )
        for ( int i = 1; i <= xsize; i++ )
            totenergy += data[ IDX(i, j) ];

    
   #undef IDX

    *energy = (double)totenergy;
    return 0;
}



