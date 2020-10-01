#include "coupling.h"
#include "couplingTypes.h"

namespace coupler {

  Array2d<double>* receive_density(const std::string cce_folder,
  		    adios2_handler &handler){
    const std::string name = "gem_density";
    return receive2d_from_ftn<double>(cce_folder,name, handler.IO, handler.eng);
  }

  void send_density(const std::string cce_folder, const Array2d<double>* density,
      adios2::IO &io, adios2::Engine &engine, adios2::Variable<double> &send_id) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const std::string fld_name = "cpl_density";
    send2d_from_C(density, cce_folder, fld_name, io, engine, send_id);
    std::cerr << rank <<  ": send " << fld_name <<" done \n";
  }

  Array2d<double>* receive_field(const std::string cce_folder,
      adios2::IO &io, adios2::Engine &eng) {
    const std::string name = "xgc_field";
    return receive2d_from_ftn<double>(cce_folder,name, io, eng);
  }

  void send_field(const std::string cce_folder, const Array2d<CV>* field,
      adios2::IO &io, adios2::Engine &engine, adios2::Variable<CV> &send_id) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const std::string fld_name = "cpl_field";
    send2d_from_C(field, cce_folder, fld_name, io, engine, send_id);
    std::cerr << rank <<  ": send " << fld_name <<" done \n";
  }

  void close_engines(adios2::Engine engine[], const int num) {
    for(int i = 0; i < num; i++) {
      engine[i].Close();
    }
  }

}
