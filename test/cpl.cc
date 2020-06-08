#include "adios2Routines.h"
#include "couplingTypes.h"
#include "dataprocess.h"
#include "importpart3mesh.h"
#include "commpart1.h"
#include "BoundaryDescr3D.h"
#include "testutilities.h"
#include <string>
#include <fstream> 

void exParFor() {
  Kokkos::parallel_for(
      4, KOKKOS_LAMBDA(const int i) {
        printf("Hello from kokkos thread i = %i\n", i);
      });
}

int main(int argc, char **argv){
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Kokkos::initialize(argc, argv);
  if(!rank) {
    printf("Hello World on Kokkos execution space %s\n",
         typeid(Kokkos::DefaultExecutionSpace).name());
    exParFor();
  }
  if(argc != 2) {
    if(!rank) printf("Usage: %s <number of timesteps>\n", argv[1]);
    exit(EXIT_FAILURE);
  }
  adios2::ADIOS adios(MPI_COMM_WORLD, adios2::DebugON);
  adios2::Variable<double> send_var[2];

  const std::string dir = "../coupling";
  const int time_step = atoi(argv[1]), RK_count = 4;

  coupler::adios2_handler gDens(adios,"gene_density");
  coupler::adios2_handler cDens(adios,"cpl_density");
  coupler::adios2_handler xFld(adios,"xgc_field");
  coupler::adios2_handler cFld(adios,"cpl_field");
  coupler::adios2_handler gRZ(adios,"gene_pproc_rz");
  coupler::adios2_handler gRX(adios,"gene_pproc_rx");
  coupler::adios2_handler gInt(adios,"gene_pproc_i");
  coupler::adios2_handler gComp(adios,"gene_pproc_c");
  coupler::adios2_handler xSurf(adios,"xgc_numsurfs");
  coupler::adios2_handler xXcoord(adios,"xgc_x_coordss");
  coupler::adios2_handler xZcoord(adios,"xgc_z_coordss");
  coupler::adios2_handler xVsurf(adios,"xgc_versurfs");
  coupler::adios2_handler cceXGC(adios,"xgc_cces");

  //receive GENE's preproc mesh discretization values
  coupler::Array1d<double>* gene_pproc_rz = coupler::receive_gene_pproc<double>(dir, gRZ);
  coupler::Array1d<double>* gene_pproc_rx = coupler::receive_gene_pproc<double>(dir, gRX);
  coupler::Array1d<int>* gene_pproc_i = coupler::receive_gene_pproc<int>(dir, gInt);

  //intialize GENE class
  const bool preproc = true;
  const bool ypar = false;
  coupler::TestCase test_case = coupler::TestCase::off;
  coupler::Part1ParalPar3D p1pp3d(gene_pproc_i->data(),gene_pproc_rx->data(),preproc);

  //receive XGC's preproc mesh discretization values
  coupler::Array1d<int>* xgc_numsurf = coupler::receive_gene_pproc<int>(dir, xSurf);
  coupler::Array1d<double>* xgc_xcoords = coupler::receive_gene_pproc<double>(dir, xXcoord);
  coupler::Array1d<double>* xgc_zcoords = coupler::receive_gene_pproc<double>(dir, xZcoord);
  coupler::Array1d<int>* xgc_versurf = coupler::receive_gene_pproc<int>(dir, xVsurf);
  coupler::Array1d<coupler::GO>* xgc_cce = coupler::receive_gene_pproc<coupler::GO>(dir, cceXGC);

  coupler::Array1d<coupler::CV>* gene_pproc_c = coupler::receive_gene_pproc<coupler::CV>(dir, gComp);
  coupler::Part3Mesh3D p3m3d(p1pp3d, xgc_numsurf->val(0), xgc_versurf->data(),xgc_cce->data(), xgc_xcoords->data(), xgc_zcoords->data(), preproc);
  const int nummode = 1;
  coupler::DatasProc3D dp3d(p1pp3d, p3m3d, preproc, test_case, ypar, nummode);
  coupler::BoundaryDescr3D bdesc(p3m3d, p1pp3d, dp3d, test_case, preproc);

  coupler::destroy(gene_pproc_rz);
  coupler::destroy(gene_pproc_rx);
  coupler::destroy(gene_pproc_i);
  coupler::destroy(gene_pproc_c);

  dp3d.InitFourierPlan3d();

  for (int i = 0; i < time_step; i++) {
    for (int j = 0; j < RK_count; j++) {
      coupler::Array2d<coupler::CV>* densityfromGENE = coupler::receive_density(dir, gDens,p1pp3d.comm_x,p1pp3d.comm_y);
      dp3d.DistriDensityRecvfromPart1(p3m3d,p1pp3d,densityfromXGC->data());
      bdesc.zDensityBoundaryBufAssign(dp3d.densin,p1pp3d);
      dp3d.InterpoDensity3D(bdesc,p3m3d,p1pp3d);
      dp3d.CmplxdataToRealdata3D();
      dp3d.AssemDensitySendtoPart3(p3m3d,p1pp3d);
      coupler::Array2d<double>* densitytoXGC = new coupler::Array2d<double>{
                                                    p3m3d.activenode,p3m3d.lj0,p3m3d.blockcount,p3m3d.lj0, 
                                                    {p3m3d.blockstart,0}};
      double** densitytmp = densitytoXGC->data();
      for(int h=0;h<p3m3d.lj0;h++){
        for(coupler::GO k=0;k<p3m3d.activenode;k++){
          densitytmp[k][h] = dp3d.denssend[k][h]; 
        }
      }
      if(p1pp3d.mype_z==0){
        coupler::send_density(dir, densitytoXGC, cDens, send_var[0]);
      }
      for(coupler::GO h=0;h<p3mp3d.activenode;h++){
        delete[] densitytmp[h];
      }
      coupler::destroy(densityfromGENE);
      coupler::destroy(densitytoXGC);

 
      coupler::Array2d<double>* fieldfromXGC = coupler::receive_field(dir, xFld,p1pp3d.comm_x,p1pp3d.comm_y);
      dp3d.DistriPotentRecvfromPart3(p3m3d,p1pp3d,fieldfromXGC->data());
      dp3d.RealdataToCmplxdata3D();
      bdesc.zPotentBoundaryBufAssign(dp3d,p3m3d,p1pp3d);
      dp3d.InterpoPotential3D(bdesc,p3m3d,p1pp3d);       
      dp3d.AssemPotentSendtoPart1(p3m3d,p1pp3d);
      coupler::Array2d<coupler::CV>* fieldtoGENE = new coupler::Array2d<coupler::CV>{
                                                   p1pp3d.totnodes,p1pp3d.lj0,p1pp3d.blockcount,
                                                   p1pp3d.lj0,{p1pp3d.blockstart,0}};
      coupler::CV** fieldtmp = fieldtoGENE->data(); 
      for(int h=0;h<p1pp3d.lj0;h++){
        for(int k=0;k<p1pp3d.totnodes;k++){
          fieldtmp[h][k] = dp3d.potentsend[h][k];
        }
      }         
      if(p1pp3d.mype_z==0){
        coupler::send_field(dir, fieldtoGENE, cFld, send_var[1]);
      }
      for(int h=0;h<p1pp3d.li0;h++){
        delete[] fieldtmp[h];
      }
      coupler::destroy(fieldtoGENE);
      coupler::destroy(fieldfromXGC)
    }
  }

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}

