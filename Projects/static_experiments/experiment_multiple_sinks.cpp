// Imports
#include <iostream>
#include <math.h> 
#include <cstdlib>
#include <fstream>
#include <typeinfo>
#include <ctime>
#include <random>
#include <vector>
#include <chrono>
#include <string>
#include <filesystem>

#include "fusion.h"

using namespace mosek::fusion;
using namespace monty;


// Auxiliar functions
auto get_C(std::vector<std::vector<double>> agents, int nmb_agents, int Px);
void get_dC(std::vector<std::vector<double>> agents, std::vector<std::vector<double>> NA, int nmb_agents, int nmb_NA, int Px,std::vector<std::vector<std::vector<double>>> (*deriv_rates_ptr));
void update_positions(std::vector<std::vector<double>> agents, std::vector<std::vector<double>> NA, int nmb_agents, int nmb_NA, int Px,int n,auto dual_vars,double lr, std::vector<std::vector<double>>(*NA_ptr));
void update_agents( std::vector<std::vector<double>> TA, std::vector<std::vector<double>> NA,int nmb_TA, int nmb_NA, std::vector<std::vector<double>> (*agents));


// Compute the link capacity for an array of agents
auto get_C(std::vector<std::vector<double>> agents, int nmb_agents, int Px)

/* Input
        -agents: Vector of R^2 positions
        -nmb_agents: Length of agents
        -Px: Transmission power

    Output: Array of size nmb_agents*nmb_agents where the element (i*nmb_agents +j) = e^(-Px * ||xi - xj||^2)
*/

{   int i,j;
    std::vector<double> max_rates(nmb_agents*nmb_agents);
    for (i=0;i<nmb_agents;i++){
        for (j=0;j<nmb_agents;j++){
            if (j!=i){
                double distance = sqrt( pow(agents[i][0] - agents[j][0], 2) + pow(agents[i][1] - agents[j][1], 2) );
                max_rates[nmb_agents*i + j] = exp(-1* Px * pow(distance,2));
            }
            else{
                max_rates[nmb_agents*i + j] = 0;
            }

        }
    }
    auto max_rates_mosek = monty::new_array_ptr<double>(max_rates);
    return max_rates_mosek;
}


// Compute the gradient of the link function w.r.t each position xi

void get_dC(std::vector<std::vector<double>> agents, std::vector<std::vector<double>> NA, int nmb_agents, int nmb_NA, int Px,std::vector<std::vector<std::vector<double>>> (*deriv_rates_ptr))
/* Input
        -agents: Vector of R^2 positions of all agents
        -NA: Vector of R^2 positions of network agents
        -nmb_agents: Length of agents
        -nmb_NA: Length of NA
        -Px: Transmission power
        -deriv_rates_ptr(by Reference): Pointer to matrix of size nmb_agents*nmb_agents where each element will be filled with dC(xi,xj)/dxi 
    
    Output: 
*/
{   int i,j;

    for (i=0; i<nmb_agents; i++){
        (*deriv_rates_ptr)[i] = std::vector<std::vector<double>>(nmb_agents);
        for (j=0;j<nmb_agents;j++){
             (*deriv_rates_ptr)[i][j] = std::vector<double>(2);
        }
    }

    for (i=nmb_agents-nmb_NA;i<nmb_agents;i++){
        for (j=0;j<nmb_agents;j++){
            double d = sqrt( pow(agents[i][0] - agents[j][0], 2) + pow(agents[i][1] - agents[j][1], 2) );
            double exponente =  1. / (1. -  2.*Px*d*d);
            if (1. >=  0.8 * 2.*Px*d*d){
                exponente = -1*(Px*2* exp(-1*Px *pow( d,2)));
            }
            (*deriv_rates_ptr)[i][j][0] = exponente *( agents[i][0] - agents[j][0]);
            (*deriv_rates_ptr)[i][j][1] = exponente * ( agents[i][1] - agents[j][1]);
            (*deriv_rates_ptr)[j][i][0] = (*deriv_rates_ptr)[i][j][0] *-1;
            (*deriv_rates_ptr)[j][i][1] = (*deriv_rates_ptr)[i][j][1] *-1;

        }
    }
}


// Function to get directions to move from dual variables and dC

void update_positions(std::vector<std::vector<double>> agents, std::vector<std::vector<double>> NA, int nmb_agents, int nmb_NA, int Px,auto dual_vars,double step_size, std::vector<std::vector<double>>(*NA_ptr))
/* Input
        -agents: Vector of R^2 positions of all agents
        -NA: Vector of R^2 positions of network agents
        -nmb_agents: Length of agents
        -nmb_NA: Length of NA
        -Px: Transmission power
        -dual_vars: List of dual variables obtained previously by solving the problem
        -step_size: Size of the step at which NA positions will be updated
        -NA_ptr(by Reference): Pointer to array of size NA*2 which will be filled with the updated NA positions
    
    Output: 
*/
{
    int i,j;
    std::vector<std::vector<std::vector<double>>> deriv_rates(nmb_agents);
    get_dC(agents,NA,nmb_agents,nmb_NA,Px,&deriv_rates);

    for (i=nmb_agents-nmb_NA;i<nmb_agents;i++){
        double direc_x = 0;
        double direc_y = 0;
        for(j=0;j<nmb_agents;j++){
            direc_x += deriv_rates[i][j][0]*dual_vars[i*nmb_agents + j] - deriv_rates[j][i][0]*dual_vars[j*nmb_agents + i];
            direc_y += deriv_rates[i][j][1]*dual_vars[i*nmb_agents + j] - deriv_rates[j][i][1]*dual_vars[j*nmb_agents + i];
        }
        (*NA_ptr)[i-(nmb_agents-nmb_NA)][0] += step_size*direc_x;
        (*NA_ptr)[i-(nmb_agents-nmb_NA)][1] += step_size*direc_y;
    }
}


// Function that from an array of task agents and network agents generates an agents array by appending both arrays

void update_agents( std::vector<std::vector<double>> TA, std::vector<std::vector<double>> NA, int nmb_TA, int nmb_NA, std::vector<std::vector<double>> (*agents)){
    /* Input:
            TA: Array of task agents
            NA: Array of network agents
            nmb_TA: Length of TA
            nmb_NA: Length of NA
            agents(Passed by reference): Pointer to array that will have TA and NA appended
        
        Output:
    */
    int i;
    for(i=0;i<nmb_TA;i++){
        (*agents)[i][0]=TA[i][0];
        (*agents)[i][1]=TA[i][1];
    }
    for(i=0;i<nmb_NA;i++){
        (*agents)[nmb_TA + i][0]=NA[i][0];
        (*agents)[nmb_TA + i][1]=NA[i][1];
    }
}




int main(int argc, char ** argv)
{
    Model::t M = new Model(); auto _M = finally([&]() { M->dispose(); });
    int i;
    int j;
    if(argc != 4){
        throw std::invalid_argument("Use: ./mosek_solution {A} {I} {N_experiments}");
    }
    const int A = std::atoi(argv[1]);
    const int I = std::atoi(argv[2]);
    const double Px =  1;
    const int nmb_agents = I+A;
    const double side = sqrt(nmb_agents);

    std::vector<std::vector<double>> TA(A);
    std::vector<std::vector<double>> NA(I);
    std::vector<std::vector<double>> agents(nmb_agents);

    // Random uniform for initialize agents
    std::uniform_real_distribution<double> unif(0,side);
    std::default_random_engine re;

    // Define variables and parameters
    auto r_ijk = M->variable( new_array_ptr<int,1>({A,A+I,A+I}), Domain::greaterThan(0.)) ;
    auto a_k = M->variable( new_array_ptr<int,1>({A})) ;
    auto C_param = M->parameter(new_array_ptr<int,1>({(A+I)*(A+I)})) ;
    auto weights_param = M->parameter(new_array_ptr<int,1>({A})) ;

    Expression::t resulted_rates;
    Expression::t constrains_network_matrix;
    
    // Link Constrain
    auto resulted_links =  Expr::sum(r_ijk, 0);
    resulted_links = Expr::reshape(resulted_links,(A+I)*(A+I));
    auto c_links = M->constraint(Expr::sub(resulted_links,C_param), Domain::lessThan(0.));

    // Network constrain (1 and 2)
    for (i=0; i<A; i++){
        Variable::t sliced_r = r_ijk->slice(new_array_ptr<int,1>({i,0,0}) , new_array_ptr<int,1>({i+1,A+I,A+I})) ;
        sliced_r = sliced_r->reshape(new_array_ptr<int,1>({A+I,A+I}));
        if (i==0){
            resulted_rates = Expr::sub( Expr::sum(sliced_r , 1) ,  Expr::sum(sliced_r,0) ) ;
        }
        else{
            resulted_rates = Expr::vstack(resulted_rates, Expr::sub( Expr::sum(sliced_r , 1) ,  Expr::sum(sliced_r,0) ));
        }
    }
    resulted_rates = Expr::reshape(resulted_rates,A,I+A);

    // 1) NA equals 0
    auto NA_resulted_rates = resulted_rates->slice( new_array_ptr<int,1>({0,A}), new_array_ptr<int,1>({A,I+A}) );
    auto c_network_agents = M->constraint(NA_resulted_rates, Domain::equalsTo(0.));

    // 2) TA equals aes

    int aux=0;
    std::vector<int> x(A*(A-1));
    for (i=0; i<A; i++){
        for (j=0;j<A;j++){
            if (i!=j){
                x[aux] = A*i + j;
                aux += 1;
            }
        }
    }

    auto stackes_aes = Expr::repeat(a_k,A,1);
    auto index_tas = monty::new_array_ptr<int>(x);

    auto TA_resulted_rates = resulted_rates->slice( new_array_ptr<int,1>({0,0}), new_array_ptr<int,1>({A,A}) );
    TA_resulted_rates = Expr::sub(TA_resulted_rates, stackes_aes);
    TA_resulted_rates = Expr::reshape(TA_resulted_rates,A*A);
    TA_resulted_rates = TA_resulted_rates->pick( index_tas );
    auto c_task_agents = M->constraint(TA_resulted_rates, Domain::greaterThan(0.));
   
    
    // Define Objective
    M->objective(ObjectiveSense::Maximize, Expr::dot(a_k,weights_param));



    // Start Running Experiments

    int nmbr_experiments = std::atoi(argv[3]);



    for(int experiment=0;experiment<nmbr_experiments;experiment++){      
        // Initialize TA agents
        for(i=0;i<A;i++){
            TA[i] = std::vector<double>(2);
            TA[i][0]=unif(re);
            TA[i][1]=unif(re);
            agents[i] = std::vector<double>(2);
            agents[i][0]=TA[i][0];
            agents[i][1]=TA[i][1];

        }

        // Initialize NA agents
        for(i=0;i<I;i++){
            NA[i] = std::vector<double>(2);
            NA[i][0]=unif(re);
            NA[i][1]=unif(re);
            agents[A + i] = std::vector<double>(2);
            agents[A + i][0]=NA[i][0];
            agents[A + i][1]=NA[i][1];

        }
        for(int sink_idx=0; sink_idx<6; sink_idx++){
            std::vector<double> W(A);
            if (sink_idx == 5){
                for (i=0; i < A; i++){
                        W[i] = 1;
                }
            }
            else{
                for (int i=0; i < A; i++){
                    if ((i+sink_idx)%5 == 0){
                        W[i] = 1;
                    }
                    else{
                        W[i] = 0;
                    }
                }
            }


            auto W_monty = monty::new_array_ptr<double>(W);

            weights_param->setValue(W_monty);
            

            std::string path="../data_multiple_sink/";
            path = path + std::to_string(A) + "TA"  +  std::to_string(I) +"NA/experiment_" + std::to_string(experiment) + "/sink_" + std::to_string(sink_idx);
            std::filesystem::create_directories(path);

            std::ofstream NA_file;
            NA_file.open(path + "/NA_positions.txt",std::ios::out);

            std::ofstream TA_file;
            TA_file.open (path + "/TA_positions.txt",std::ios::out);

            std::ofstream performance_file;
            performance_file.open (path + "/performance.txt",std::ios::out);

            std::ofstream iterations_file;
            iterations_file.open (path + "/iterations_data.txt",std::ios::out);
            


            // Output TA positions to file
            for (i=0;i<A;i++){
                TA_file << TA[i][0] << ", " << TA[i][1];
                if(i!=A-1){
                    TA_file << ", ";
                }
            }
            TA_file << "\n";
            TA_file.close();

            // Loop of solving MCFP and updating positions
            double lr_0;
            double decay;
            if(sink_idx == 5){
                lr_0 = 0.4;
                decay = 0.97;
            }
            else{
                lr_0 = 0.8;
                decay = 0.97;
            }
            
            double lr=lr_0;
            int max_iter = 100;
            int iter=0;
            bool converged = false;
            float tol = 1e-4;
            float last_primal = -1;

            uint64_t ms_0 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            while(iter<max_iter){// & converged == false){

                // Get value of C
                auto C = get_C(agents,nmb_agents,Px);
                C_param->setValue(C);

                // Solve problem
                
                M->solve();
                auto duals = *(c_links->dual());

                // Update NA positons
                lr = lr*decay;
                update_positions(agents, NA, nmb_agents, I, Px,duals,lr,&NA);
                update_agents(TA,NA,A,I,&agents);

                // Dump to file NA positions
                for (i=0;i<I;i++){
                    NA_file << NA[i][0] << ", " << NA[i][1];
                    if(i!=I-1){
                        NA_file << ", ";
                    }
                }
                NA_file << "\n";

                // Dump to file the performance
                performance_file <<  M->primalObjValue() << "\n";
                
                // Update iteration and check stop condition
                iter++;
                converged =  abs(M->primalObjValue() - last_primal) < tol;
                last_primal =  M->primalObjValue();
                }
            uint64_t ms_f = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            
            //iterations_file << iter << '\n' << ms_f-ms_0 << '\n';
            iterations_file << ms_f-ms_0 << '\n';
            iterations_file.close();
            NA_file.close();
            performance_file.close();
            std::system("clear");
            std::cout<< experiment <<"/" << nmbr_experiments << "\n";

        }
    }
}
