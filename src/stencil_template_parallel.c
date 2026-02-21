/*

/*
 *
 *  mysizex   :   local x-extendion of your patch
 *  mysizey   :   local y-extension of your patch
 *
 */


#include "stencil_template_parallel.h"



// ------------------------------------------------------------------
// ------------------------------------------------------------------

int main(int argc, char **argv)
{
	MPI_Comm myCOMM_WORLD;
	int  Rank, Ntasks;		// Node ID, Total processes
	int  neighbours[4];		// Node's N/S/E/W neighbors

	int  Niterations;
	int  periodic;

	// S: Global plate size [total_x, total_y]
	// N: MPI task grid dimensions [Nx, Ny] - how we split the domain
	vec2_t S, N;
	
	int      Nsources;
	int      Nsources_local;	// how many sources THIS task owns
	vec2_t  *Sources_local;		// coordinates of THIS task's sources
	double   energy_per_source;

	plane_t   planes[2];

	// Two sets of 4 buffers each:
	// buffers[SEND][NORTH], ..., buffers[SEND][WEST]  |  buffers[RECEIVE][NORTH], ..., buffers[RECEIVE][WEST]
	buffers_t buffers[2];
	
	int output_energy_stat_perstep;
	
	/* initialize MPI envrionment 
	- MPI_COMM_WORLD: The global "phone network" connecting all processes
	- MPI_Comm_rank: "What's my phone number?" (0, 1, 2, ...)
	- MPI_Comm_size: "How many phones are in the network?"
	- MPI_Comm_dup: "Give me my own private line" (prevents communication interference)
	*/
	{
		int level_obtained;
		
		// NOTE: change MPI_FUNNELED if appropriate
		//
		// Starts MPI, but also tells MPI we want to use threads (omp)
		MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &level_obtained );

		// MPI_THREAD_FUNNELED: Thread safety level - means "only the main thread will make MPI calls." 
		// This is the simplest safe mode when combining MPI + OpenMP
		if ( level_obtained < MPI_THREAD_FUNNELED ) 
		{	// If MPI can't provide the threading support we need, abort.
			printf( "MPI_thread level obtained is %d instead of %d\n", level_obtained, MPI_THREAD_FUNNELED );
			MPI_Finalize();
			exit(1); 
		}
		
		MPI_Comm_rank(MPI_COMM_WORLD, &Rank);          // "What is my process ID?"
		MPI_Comm_size(MPI_COMM_WORLD, &Ntasks);        // "How many total processes?"
		MPI_Comm_dup (MPI_COMM_WORLD, &myCOMM_WORLD);  // Make a private copy of the communicator
	}
	
	
	/* argument checking and setting */
	int ret = initialize ( 
		&myCOMM_WORLD, 
		Rank, Ntasks, 
		argc, argv, 
		&S, &N, 
		&periodic, &output_energy_stat_perstep,
		neighbours, &Niterations,
		&Nsources, &Nsources_local, 
		&Sources_local, &energy_per_source,
		&planes[0], &buffers[0] 
	);

	if ( ret )
	{
		printf("task %d is opting out with termination code %d\n", Rank, ret );
		
		MPI_Finalize();
		return 0;
	}
	
	
	int current = OLD;
	double t1 = MPI_Wtime();   /* take wall-clock time. Used for performance measurement.*/
	
	for (int iter = 0; iter < Niterations; ++iter)
	{
		
		// 8 "receipts" for non-blocking MPI operations. 
		// You start the operation, get a receipt, and later check if it's done.
		MPI_Request reqs[8];
		
		/* new energy from sources */
		// Serial injected at ALL sources. Parallel only injects at THIS task's local sources
		inject_energy( periodic, Nsources_local, Sources_local, energy_per_source, &planes[current], N );


		/* -------------------------------------- */
		{
		double *data = planes[current].data;
		uint xsize = planes[current].size[_x_];
		uint ysize = planes[current].size[_y_];
		uint fxsize = xsize + 2;

		#define IDX(i, j) ((j) * fxsize + (i))

		// [A] Pack send buffers: copy edge rows/columns into flat buffers

		// NORTH: first interior row (j=1)
		for (uint k = 0; k < xsize; k++)
			buffers[SEND][NORTH][k] = data[IDX(k + 1, 1)];

		// SOUTH: last interior row (j=ysize)
		for (uint k = 0; k < xsize; k++)
			buffers[SEND][SOUTH][k] = data[IDX(k + 1, ysize)];

		// EAST: last interior column (i=xsize)
		for (uint k = 0; k < ysize; k++)
			buffers[SEND][EAST][k] = data[IDX(xsize, k + 1)];

		// WEST: first interior column (i=1)
		for (uint k = 0; k < ysize; k++)
			buffers[SEND][WEST][k] = data[IDX(1, k + 1)];

		// [B] Non-blocking halo exchange using Isend/Irecv
		//     Tag convention: send uses sender's direction as tag,
		//     receive uses opposite direction as tag.
		//     e.g. I send NORTH with tag=NORTH; my north neighbor
		//     receives from SOUTH with tag=NORTH. Tags match.
		int nreqs = 0;

		MPI_Isend(buffers[SEND][NORTH], xsize, MPI_DOUBLE,
		          neighbours[NORTH], NORTH, myCOMM_WORLD, &reqs[nreqs++]);
		MPI_Irecv(buffers[RECV][NORTH], xsize, MPI_DOUBLE,
		          neighbours[NORTH], SOUTH, myCOMM_WORLD, &reqs[nreqs++]);

		MPI_Isend(buffers[SEND][SOUTH], xsize, MPI_DOUBLE,
		          neighbours[SOUTH], SOUTH, myCOMM_WORLD, &reqs[nreqs++]);
		MPI_Irecv(buffers[RECV][SOUTH], xsize, MPI_DOUBLE,
		          neighbours[SOUTH], NORTH, myCOMM_WORLD, &reqs[nreqs++]);

		MPI_Isend(buffers[SEND][EAST], ysize, MPI_DOUBLE,
		          neighbours[EAST], EAST, myCOMM_WORLD, &reqs[nreqs++]);
		MPI_Irecv(buffers[RECV][EAST], ysize, MPI_DOUBLE,
		          neighbours[EAST], WEST, myCOMM_WORLD, &reqs[nreqs++]);

		MPI_Isend(buffers[SEND][WEST], ysize, MPI_DOUBLE,
		          neighbours[WEST], WEST, myCOMM_WORLD, &reqs[nreqs++]);
		MPI_Irecv(buffers[RECV][WEST], ysize, MPI_DOUBLE,
		          neighbours[WEST], EAST, myCOMM_WORLD, &reqs[nreqs++]);

		MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

		// [C] Unpack received data into ghost cells
		if (neighbours[NORTH] != MPI_PROC_NULL)
			for (uint k = 0; k < xsize; k++)
				data[IDX(k + 1, 0)] = buffers[RECV][NORTH][k];

		if (neighbours[SOUTH] != MPI_PROC_NULL)
			for (uint k = 0; k < xsize; k++)
				data[IDX(k + 1, ysize + 1)] = buffers[RECV][SOUTH][k];

		if (neighbours[EAST] != MPI_PROC_NULL)
			for (uint k = 0; k < ysize; k++)
				data[IDX(xsize + 1, k + 1)] = buffers[RECV][EAST][k];

		if (neighbours[WEST] != MPI_PROC_NULL)
			for (uint k = 0; k < ysize; k++)
				data[IDX(0, k + 1)] = buffers[RECV][WEST][k];

		#undef IDX
		}
		/* --------------------------------------  */
		/* update grid points */
		
		update_plane( periodic, N, &planes[current], &planes[!current] );

		/* output if needed */
		if ( output_energy_stat_perstep )
		output_energy_stat ( iter, &planes[!current], (iter+1) * Nsources*energy_per_source, Rank, &myCOMM_WORLD );
		
		/* swap plane indexes for the new iteration */
		current = !current;	
	}
	
	t1 = MPI_Wtime() - t1;

	if (Rank == 0)
		printf("Elapsed time: %f seconds\n", t1);

	output_energy_stat ( -1, &planes[!current], Niterations * Nsources*energy_per_source, Rank, &myCOMM_WORLD );
	
	memory_release( buffers, planes );
	
	MPI_Finalize();

	return 0;
}


/* ==========================================================================
   =                                                                        =
   =   routines called within the integration loop                          =
   ========================================================================== */





/* ==========================================================================
   =                                                                        =
   =   initialization                                                       =
   ========================================================================== */


uint simple_factorization( uint, int *, uint ** );

int initialize_sources( 
	int       ,
	int       ,
	MPI_Comm *,
	uint   [2],
	int       ,
	int      *,
	vec2_t  ** 
);


int memory_allocate ( 
	const int       *,
    const vec2_t     ,
    buffers_t       *,
    plane_t         * 
);
		      

int initialize ( 
	MPI_Comm  *Comm,
	int        Me,                  //   NEW: the rank of the calling process
	int        Ntasks,              //   NEW :the total number of MPI ranks
	int        argc,                // the argc from command line
	char     **argv,                // the argv from command line
	vec2_t    *S,                   // the size of the plane
	vec2_t    *N,                   //   NEW: two-uint array defining the MPI tasks' grid
	int       *periodic,            // periodic-boundary tag
	int       *output_energy_stat,
	int       *neighbours,          //   NEW: four-int array that gives back the neighbours of the calling task
	int       *Niterations,         // how many iterations
	int       *Nsources,            // how many heat sources
	int       *Nsources_local,		//   NEW: sources on this task
	vec2_t   **Sources_local,		//   NEW: local source coords
	double    *energy_per_source,   // how much heat per source
	plane_t   *planes,
	buffers_t *buffers				//   NEW: comm buffers
)
{
	int halt = 0;
	int ret;
	int verbose = 0;
	
	// ··································································
	// set default values

	// Note the (*S)[_x_] syntax: `S` is a `vec2_t *` (pointer to a `vec2_t`). 
	// So *S dereferences the pointer to get the vec2_t, then `[_x_]` indexes into it. 
	// Parentheses are needed because `[]` has higher precedence than `*`.
	(*S)[_x_]         = 10000;
	(*S)[_y_]         = 10000;
	
	*periodic         = 0;
	*Nsources         = 4;
	*Nsources_local   = 0;
	*Sources_local    = NULL;
	*Niterations      = 1000;
	*energy_per_source = 1.0;

	if ( planes == NULL ) {
		perror("Error: NULL pointer passed for 'planes'. Exiting");
		exit(1);
	}

	// bug?								  [0]	
	planes[OLD].size[0] = planes[OLD].size[1] = 0;
	planes[NEW].size[0] = planes[NEW].size[1] = 0;
	
	for ( int i = 0; i < 4; i++ )
		neighbours[i] = MPI_PROC_NULL;

	for ( int b = 0; b < 2; b++ )
		for ( int d = 0; d < 4; d++ )
		buffers[b][d] = NULL;
	
	// ··································································
	// process the commadn line
	// 
	while ( 1 )
	{
		int opt;
		while((opt = getopt(argc, argv, ":hx:y:e:E:n:o:p:v:")) != -1)
		{
		switch( opt )
		{
		case 'x': (*S)[_x_] = (uint)atoi(optarg);
			break;

		case 'y': (*S)[_y_] = (uint)atoi(optarg);
			break;

		case 'e': *Nsources = atoi(optarg);
			break;

		case 'E': *energy_per_source = atof(optarg);
			break;

		case 'n': *Niterations = atoi(optarg);
			break;

		case 'o': *output_energy_stat = (atoi(optarg) > 0);
			break;

		case 'p': *periodic = (atoi(optarg) > 0);
			break;

		case 'v': verbose = atoi(optarg);
			break;

		case 'h': {
			if ( Me == 0 )
			printf( "\nvalid options are ( values btw [] are the default values ):\n"
				"-x    x size of the plate [10000]\n"
				"-y    y size of the plate [10000]\n"
				"-e    how many energy sources on the plate [4]\n"
				"-E    how many energy sources on the plate [1.0]\n"
				"-n    how many iterations [1000]\n"
				"-p    whether periodic boundaries applies  [0 = false]\n\n"
				);
			halt = 1; }
			break;
			
			
		case ':': printf( "option -%c requires an argument\n", optopt);
			break;
			
		case '?': printf(" -------- help unavailable ----------\n");
			break;
		}
		}

		if ( opt == -1 )
		break;
	}

	if ( halt )
		return 1;
	
	
	// ··································································
	/*
	* here we should check for all the parms being meaningful
	*
	*/

	// ...

	
	// ··································································
	/*
	* find a suitable domain decomposition
	* very simple algorithm, you may want to
	* substitute it with a better one
	*
	* the plane Sx x Sy will be solved with a grid
	* of Nx x Ny MPI tasks
	*/

	vec2_t Grid;
	double formfactor = ((*S)[_x_] >= (*S)[_y_] ? (double)(*S)[_x_]/(*S)[_y_] : (double)(*S)[_y_]/(*S)[_x_] );
	int    dimensions = 2 - (Ntasks <= ((int)formfactor+1) );

	// 1D decomposition
	if ( dimensions == 1 )
		{
		if ( (*S)[_x_] >= (*S)[_y_] )
			Grid[_x_] = Ntasks, Grid[_y_] = 1;
		else
			Grid[_x_] = 1, Grid[_y_] = Ntasks;
		}
	// 2D decomposition
	else
		{
		int   Nf;
		uint *factors;
		uint  first = 1;
		ret = simple_factorization( Ntasks, &Nf, &factors );
		
		for ( int i = 0; (i < Nf) && ((Ntasks/first)/first > formfactor); i++ )
			first *= factors[i];

		if ( (*S)[_x_] > (*S)[_y_] )
			Grid[_x_] = Ntasks/first, Grid[_y_] = first;
		else
			Grid[_x_] = first, Grid[_y_] = Ntasks/first;
		}

	(*N)[_x_] = Grid[_x_];
	(*N)[_y_] = Grid[_y_];
	

	// ··································································
	// my cooridnates in the grid of processors
	//
	int X = Me % Grid[_x_];
	int Y = Me / Grid[_x_];

	// ··································································
	// find my neighbours
	//


	// If periodic, tasks at edge have MPI_PROC_NULL (no neighbor)
	if ( Grid[_x_] > 1 )
	{  
		if ( *periodic ) {       
			neighbours[EAST]  = Y*Grid[_x_] + (Me + 1 ) % Grid[_x_];
			neighbours[WEST]  = (X%Grid[_x_] > 0 ? Me-1 : (Y+1)*Grid[_x_]-1); 
		}
		
		else {
			neighbours[EAST]  = ( X < Grid[_x_]-1 ? Me+1 : MPI_PROC_NULL );
			neighbours[WEST]  = ( X > 0 ? (Me-1)%Ntasks : MPI_PROC_NULL ); 
		}  
	}

	if ( Grid[_y_] > 1 )
	{
		if ( *periodic ) {      
			neighbours[NORTH] = (Ntasks + Me - Grid[_x_]) % Ntasks;
			neighbours[SOUTH] = (Ntasks + Me + Grid[_x_]) % Ntasks; 
		}

		else {    
			neighbours[NORTH] = ( Y > 0 ? Me - Grid[_x_]: MPI_PROC_NULL );
			neighbours[SOUTH] = ( Y < Grid[_y_]-1 ? Me + Grid[_x_] : MPI_PROC_NULL ); 
		}
	}

	// ··································································
	// the size of my patch
	//

	/*
	* every MPI task determines the size sx x sy of its own domain
	* REMIND: the computational domain will be embedded into a frame
	*         that is (sx+2) x (sy+2)
	*         the outern frame will be used for halo communication or
	*/
	
	// Dividing work unevenly if there is a remainder
	// The first `r` tasks get one extra column
	vec2_t mysize;
	uint s = (*S)[_x_] / Grid[_x_];
	uint r = (*S)[_x_] % Grid[_x_];
	mysize[_x_] = s + (X < r);
	s = (*S)[_y_] / Grid[_y_];
	r = (*S)[_y_] % Grid[_y_];
	mysize[_y_] = s + (Y < r);

	// Store local patch size in both plane structs
	planes[OLD].size[0] = mysize[0];
	planes[OLD].size[1] = mysize[1];
	planes[NEW].size[0] = mysize[0];
	planes[NEW].size[1] = mysize[1];
	

	if ( verbose > 0 )
	{
		if ( Me == 0 ) {
			printf("Tasks are decomposed in a grid %d x %d\n\n", Grid[_x_], Grid[_y_] );
			fflush(stdout);
		}

		MPI_Barrier(*Comm);
		
		for ( int t = 0; t < Ntasks; t++ )
		{
			if ( t == Me )
			{
				printf("Task %4d :: "
					"\tgrid coordinates : %3d, %3d\n"
					"\tneighbours: N %4d    E %4d    S %4d    W %4d\n",
					Me, X, Y,
					neighbours[NORTH], neighbours[EAST],
					neighbours[SOUTH], neighbours[WEST] );
				fflush(stdout);
			}
			MPI_Barrier(*Comm);
		}
	}

	
	// ··································································
	// allocae the needed memory
	//
	ret = memory_allocate( neighbours, *N, buffers, planes );
	

	// ··································································
	// allocae the heat sources
	//
	ret = initialize_sources( Me, Ntasks, Comm, mysize, *Nsources, Nsources_local, Sources_local );
	
	
	return 0;  
}


uint simple_factorization( uint A, int *Nfactors, uint **factors )
/*
 * rought factorization;
 * assumes that A is small, of the order of <~ 10^5 max,
 * since it represents the number of tasks
 #
 */
{
	int N = 0;
	int f = 2;
	uint _A_ = A;

	while ( f < A )
	{
		while( _A_ % f == 0 ) 
		{
			N++;
			_A_ /= f; 
		}
		f++;
	}

	*Nfactors = N;
	uint *_factors_ = (uint*)malloc( N * sizeof(uint) );

	N   = 0;
	f   = 2;
	_A_ = A;

	while ( f < A )
	{
		while( _A_ % f == 0 ) 
		{
			_factors_[N++] = f;
			_A_ /= f; 
		}
		f++;
	}

	*factors = _factors_;
	return 0;
}


int initialize_sources( 
	int       Me,
	int       Ntasks,
	MPI_Comm *Comm,
	vec2_t    mysize,
	int       Nsources,
	int      *Nsources_local,
	vec2_t  **Sources 
)
{
	// Task 0 randomly assigns source locations to every task

	srand48(time(NULL) ^ Me);
	int *tasks_with_sources = (int*)malloc( Nsources * sizeof(int) );
	
	if ( Me == 0 )
		{
		for ( int i = 0; i < Nsources; i++ )
			tasks_with_sources[i] = (int)lrand48() % Ntasks;
		}
	
	MPI_Bcast( tasks_with_sources, Nsources, MPI_INT, 0, *Comm );

	int nlocal = 0;
	for ( int i = 0; i < Nsources; i++ )
		nlocal += (tasks_with_sources[i] == Me);
	*Nsources_local = nlocal;
	
	if ( nlocal > 0 )
		{
		vec2_t * restrict helper = (vec2_t*)malloc( nlocal * sizeof(vec2_t) );      
		for ( int s = 0; s < nlocal; s++ )
		{
		helper[s][_x_] = 1 + lrand48() % mysize[_x_];
		helper[s][_y_] = 1 + lrand48() % mysize[_y_];
		}

		*Sources = helper;
		}
	
	free( tasks_with_sources );

	return 0;
}



int memory_allocate ( 
	const int   *neighbours,
	const vec2_t N,
	buffers_t   *buffers_ptr,
	plane_t     *planes_ptr
)
{
	/*
	here you allocate the memory buffers that you need to
	(i)  hold the results of your computation
	(ii) communicate with your neighbours

	The memory layout that I propose to you is as follows:

	(i) --- calculations
	you need 2 memory regions: the "OLD" one that contains the
	results for the step (i-1)th, and the "NEW" one that will contain
	the updated results from the step ith.

	Then, the "NEW" will be treated as "OLD" and viceversa.

	These two memory regions are indexed by *plate_ptr:

	plane_ptr[0] ==> the "OLD" region
	plane_ptr[1] ==> the "NEW" region


	(ii) --- communications

	you may need two buffers (one for sending and one for receiving)
	for each one of your neighnours, that are at most 4:
	north, south, east amd west.      

	To them you need to communicate at most mysizex or mysizey double data.

	These buffers are indexed by the buffer_ptr pointer so that

	(*buffers_ptr)[SEND][ {NORTH,...,WEST} ] = .. some memory regions
	(*buffers_ptr)[RECV][ {NORTH,...,WEST} ] = .. some memory regions
	
	--->> Of course you can change this layout as you prefer
	
	*/

	if (planes_ptr == NULL )
	{
		perror("Error: NULL pointer passed for 'planes'. Exiting");
		exit(1);
	}

	if (buffers_ptr == NULL )
	{
		perror("Error: NULL pointer passed for 'buffers'. Exiting");
		exit(1);
	}
	
	// ··················································
	// allocate memory for data
	// we allocate the space needed for the plane plus a contour frame
	// that will contains data form neighbouring MPI tasks
	unsigned int frame_size = (planes_ptr[OLD].size[_x_]+2) * (planes_ptr[OLD].size[_y_]+2);

	planes_ptr[OLD].data = (double*)malloc( frame_size * sizeof(double) );
	if ( planes_ptr[OLD].data == NULL )
		// manage the malloc fail
		;
	memset ( planes_ptr[OLD].data, 0, frame_size * sizeof(double) );

	planes_ptr[NEW].data = (double*)malloc( frame_size * sizeof(double) );
	if ( planes_ptr[NEW].data == NULL )
		// manage the malloc fail
		;
	memset ( planes_ptr[NEW].data, 0, frame_size * sizeof(double) );


	// ··················································
	// buffers for north and south communication 
	// are not really needed
	//
	// in fact, they are already contiguous, just the
	// first and last line of every rank's plane
	//
	// you may just make some pointers pointing to the
	// correct positions
	//

	// or, if you prefer, just go on and allocate buffers
	// also for north and south communications

	// ··················································
	// allocate buffers
	//
	{
		uint xsize = planes_ptr[OLD].size[_x_];
		uint ysize = planes_ptr[OLD].size[_y_];

		for (int dir = 0; dir < 4; dir++)
		{
			// N/S exchange full rows (xsize elements), E/W exchange full columns (ysize elements)
			uint bufsize = (dir == NORTH || dir == SOUTH) ? xsize : ysize;

			buffers_ptr[SEND][dir] = (double *)malloc(bufsize * sizeof(double));
			buffers_ptr[RECV][dir] = (double *)malloc(bufsize * sizeof(double));

			memset(buffers_ptr[SEND][dir], 0, bufsize * sizeof(double));
			memset(buffers_ptr[RECV][dir], 0, bufsize * sizeof(double));
		}
	}

	// ··················································

	
	return 0;
}



int memory_release ( buffers_t *buffers, plane_t *planes)
{
	// Free communication buffers
	if ( buffers != NULL )
	{
		for (int b = 0; b < 2; b++)          // SEND and RECV
			for (int d = 0; d < 4; d++)      // NORTH, SOUTH, EAST, WEST
				if ( buffers[b][d] != NULL )
				{
					free( buffers[b][d] );
					buffers[b][d] = NULL;	 // defensive: prevent double-free
				}
	}

	// Free plane data
	if ( planes != NULL )
	{
		if ( planes[OLD].data != NULL )
			free( planes[OLD].data );

		if ( planes[NEW].data != NULL )
			free( planes[NEW].data );
	}

	return 0;
}


int output_energy_stat ( 
	int       step, 
	plane_t  *plane, 
	double    budget, 
	int       Me, 
	MPI_Comm *Comm 
)
{
	double system_energy = 0;
	double tot_system_energy = 0;
	get_total_energy ( plane, &system_energy );
	
	MPI_Reduce ( &system_energy, &tot_system_energy, 1, MPI_DOUBLE, MPI_SUM, 0, *Comm );
	
	if ( Me == 0 )
		{
		if ( step >= 0 )
		printf(" [ step %4d ] ", step ); fflush(stdout);

		
		printf( "total injected energy is %g, "
			"system energy is %g "
			"( in avg %g per grid point)\n",
			budget,
			tot_system_energy,
			tot_system_energy / (plane->size[_x_]*plane->size[_y_]) );
		}
	
	return 0;
}
