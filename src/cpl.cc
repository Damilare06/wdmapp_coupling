#include "coupling.h"
#include "couplingTypes.h"

//void exParFor() {
//  Kokkos::parallel_for(
//      4, KOKKOS_LAMBDA(const int i) {
//        printf("Hello from kokkos thread i = %i\n", i);
//      });
//}

int main(int argc, char **argv){
  int rank, nprocs;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  //Kokkos::initialize(argc, argv);
  //if(!rank) {
  //  printf("Hello World on Kokkos execution space %s\n",
  //       typeid(Kokkos::DefaultExecutionSpace).name());
  //  exParFor();
  //}

  if(argc != 2) {
    if(!rank) printf("Usage: %s <number of timesteps>\n", argv[1]);
    exit(EXIT_FAILURE);
  }
  const std::string dir = "../coupling";
  const int time_step = 1, RK_count = 4;

  fprintf(stderr, "%d number of time steps: %d\n", rank, time_step);

  const std::string xmlfile = "adios2cfg.xml";
  adios2::ADIOS adios(MPI_COMM_WORLD, adios2::DebugON);
  adios2::Variable<double> senddensity;
  adios2::Variable<coupler::CV> sendfield;

  coupler::adios2_handler gDens(adios,"gem_density");
  coupler::adios2_handler cDens(adios,"cpl_density");
  coupler::adios2_handler xFld(adios,"xgc_field");
  coupler::adios2_handler cFld(adios,"cpl_field");
  coupler::adios2_handler gCy(adios,"gem_cy_array");

  //receive GENE's preproc mesh discretization values
  //coupler::Array1d<int>* gem_pproc = coupler::receive_gem_pproc<int>(dir, gCy.IO, gCy.eng);
  //coupler::Array1d*<double> xgc_pproc = coupler::receive_xgc_pproc(dir, IO[1], eng[1]);

  //Perform coupling routines
  //coupler::destroy(gem_pproc);

  for (int i = 0; i < time_step; i++) {
    for (int j = 0; j < RK_count; j++) {
      coupler::Array2d<double>* density = coupler::receive_density(dir, gDens);
      coupler::printSomeDensityVals(density);
      coupler::send_density(dir, density, cDens.IO, cDens.eng, senddensity);
      coupler::destroy(density);

      coupler::Array2d<double>* field = coupler::receive_field(dir, xFld.IO, xFld.eng);
      coupler::Array2d<coupler::CV>* field_CV = new coupler::Array2d<coupler::CV>(10,10,10,10,10);
      coupler::send_field(dir, field_CV, cFld.IO, cFld.eng, sendfield);
      coupler::destroy(field);
    }
  }

  gCy.close();
  gDens.close();
  cDens.close();
  xFld.close();
  cFld.close();

  //Kokkos::finalize();
  MPI_Finalize();
  return 0;
}

