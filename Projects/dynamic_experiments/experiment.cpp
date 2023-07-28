// Imports

#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <thread>
#include <math.h> 
#include "fusion.h"
#include <fstream>
#include <ctime>
#include <typeinfo>
#include <filesystem>

using namespace mosek::fusion;
using namespace monty;


// Auxiliar functions
auto get_C(std::vector<std::vector<double>> agents, int nmb_agents, int d_connected);
void get_dC(std::vector<std::vector<double>> agents, std::vector<std::vector<double>> NA, int nmb_agents, int nmb_NA, int d_connected,std::vector<std::vector<std::vector<double>>> (*deriv_rates_ptr));
void update_positions(std::vector<std::vector<double>> agents, std::vector<std::vector<double>> NA, int nmb_agents, int nmb_NA, int d_connected,auto dual_vars,double lr, std::vector<std::vector<double>>(*NA_ptr));
void update_agents( std::vector<std::vector<double>> TA, std::vector<std::vector<double>> NA,int nmb_TA, int nmb_NA, std::vector<std::vector<double>> (*agents));
void get_directions(std::vector<std::vector<double>> agents, std::vector<std::vector<double>> NA, int nmb_agents, int nmb_NA, int d_connected,auto dual_vars,double max_delta, std::vector<std::vector<double>>(*NA_directions_ptr));
void simulate(std::vector<std::vector<double>> (*TA_ptr), std::vector<std::vector<double>> (*NA_ptr),int nmbr_TA, int nmbr_NA, std::vector<std::vector<double>> (*NA_directions_ptr),double sim_time,double deltaT, bool *finished_ptr,std::string path);


// Compute the link capacity for an array of agents
auto get_C(std::vector<std::vector<double>> agents, int nmb_agents, int d_connected)
/* Input
        -agents: Vector of R^2 positions
        -nmb_agents: Length of agents
        -d_connected: Distance where link is 1/e

    Output: Array of size nmb_agents*nmb_agents where the element (i*nmb_agents +j) = e^(- (||xi - xj|| / d_connected)^2)
*/

{   int i,j;
    std::vector<double> max_rates(nmb_agents*nmb_agents);
    for (i=0;i<nmb_agents;i++){
        for (j=0;j<nmb_agents;j++){
            if (j!=i){
                double d = sqrt( pow(agents[i][0] - agents[j][0], 2) + pow(agents[i][1] - agents[j][1], 2) );
                max_rates[nmb_agents*i + j] = exp(-1*pow(d/d_connected,2));

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

void get_dC(std::vector<std::vector<double>> agents, std::vector<std::vector<double>> NA, int nmb_agents, int nmb_NA, int d_connected,std::vector<std::vector<std::vector<double>>> (*deriv_rates_ptr))
/* Input
        -agents: Vector of R^2 positions of all agents
        -NA: Vector of R^2 positions of network agents
        -nmb_agents: Length of agents
        -nmb_NA: Length of NA
        -d_connected: Distance where link is 1/e
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
            double exponente =  0.895 / (1. -  2.*pow(d/d_connected,2));
            if (d <=  1.22 * d_connected / sqrt(2) ){
                exponente = -1*(2/d_connected/d_connected * exp(-1*pow(d/d_connected,2)));
            }
            (*deriv_rates_ptr)[i][j][0] = exponente *( agents[i][0] - agents[j][0]);
            (*deriv_rates_ptr)[i][j][1] = exponente * ( agents[i][1] - agents[j][1]);
            (*deriv_rates_ptr)[j][i][0] = (*deriv_rates_ptr)[i][j][0] *-1;
            (*deriv_rates_ptr)[j][i][1] = (*deriv_rates_ptr)[i][j][1] *-1;

        }
    }
}



// Function to get directions to move from dual variables and dC

void update_positions(std::vector<std::vector<double>> agents, std::vector<std::vector<double>> NA, int nmb_agents, int nmb_NA, int d_connected,auto dual_vars,double lr, std::vector<std::vector<double>>(*NA_ptr))
/* Input
        -agents: Vector of R^2 positions of all agents
        -NA: Vector of R^2 positions of network agents
        -nmb_agents: Length of agents
        -nmb_NA: Length of NA
        -d_connected: Distance where link is 1/e
        -dual_vars: List of dual variables obtained previously by solving the problem
        -step_size: Size of the step at which NA positions will be updated
        -NA_ptr(by Reference): Pointer to array of size NA*2 which will be filled with the updated NA positions
    
    Output: 
*/
{    
    int i,j;
    std::vector<std::vector<std::vector<double>>> deriv_rates(nmb_agents);
    get_dC(agents,NA,nmb_agents,nmb_NA,d_connected,&deriv_rates);

    for (i=nmb_agents-nmb_NA;i<nmb_agents;i++){
        double direc_x = 0;
        double direc_y = 0;
        for(j=0;j<nmb_agents;j++){
            direc_x += deriv_rates[i][j][0]*dual_vars[i*nmb_agents + j] - deriv_rates[j][i][0]*dual_vars[j*nmb_agents + i];
            direc_y += deriv_rates[i][j][1]*dual_vars[i*nmb_agents + j] - deriv_rates[j][i][1]*dual_vars[j*nmb_agents + i];
        }
        (*NA_ptr)[i-(nmb_agents-nmb_NA)][0] += lr*direc_x;
        (*NA_ptr)[i-(nmb_agents-nmb_NA)][1] += lr*direc_y;
    }
}

// Function that from an array of task agents and network agents generates an agents array by appending both arrays

void update_agents( std::vector<std::vector<double>> TA, std::vector<std::vector<double>> NA, int nmb_TA, int nmb_NA, std::vector<std::vector<double>> (*agents))
    /* Input:
            TA: Array of task agents
            NA: Array of network agents
            nmb_TA: Length of TA
            nmb_NA: Length of NA
            agents(Passed by reference): Pointer to array that will have TA and NA appended
        
        Output:
    */
{
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


// Thread that will simulate the update of the positions every deltaT seconds

void simulate(std::vector<std::vector<double>> (*TA_ptr), std::vector<std::vector<double>> (*NA_ptr),int nmbr_TA, int nmbr_NA, std::vector<std::vector<double>> (*NA_directions_ptr),double sim_time,double deltaT, bool *finished_ptr, std::string path)
    /* Input:
            TA_ptr(by Reference): pointer to Array of task agents (shared with main Thread)
            NA_ptr(by Reference): pointer to Array of network agents (shared with main Thread)
            nmb_TA: Length of TA
            nmb_NA: Length of NA
            NA_directions_ptr: pointer to Array of direction to update agents (shared with main Thread)
            NA_max_velocity: Maximum velocity NA can move
            sim_time: Duration of simulation
            deltaT: Time it takes to update positions
            finished_ptr: pointer to boolean variable that let now main Thread when simulation is over
        Output:
    */
{    
    std::ofstream NA_file;
    NA_file.open (path + "/NA_positions.txt",std::ios::out);
    std::ofstream TA_file;
    TA_file.open (path + "/TA_positions.txt",std::ios::out);   
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal{0, 0.1};

    std::vector<std::vector<double>> local_TA = (*TA_ptr) ;
    std::vector<std::vector<double>> local_NA = (*NA_ptr);
    std::vector<std::vector<double>> local_NA_directions = (*NA_directions_ptr);
    std::vector<std::vector<double>> local_TA_velocity(nmbr_TA);
    double accel_x;
    double accel_y;
    double decay = 1;
    double radius = sqrt((nmbr_TA+nmbr_NA)/3.1416);
    for(int i = 0;i<nmbr_TA;i++){
        local_TA_velocity[i] = std::vector<double>(2);
        local_TA_velocity[i][0] = 0;
        local_TA_velocity[i][1] = 0;
    }

    for (int iter_t=0; iter_t < sim_time/deltaT;iter_t++){
        uint64_t ms_i = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Update TA positions
        local_TA = (*TA_ptr);
        local_NA = (*NA_ptr);
        local_NA_directions = (*NA_directions_ptr);
        decay *= 0.95;
        for (int i=0; i<nmbr_TA;i++){
            accel_x = normal(gen)*decay;
            accel_y = normal(gen)*decay;
            local_TA_velocity[i][0] += accel_x*deltaT;
            local_TA_velocity[i][1] += accel_y*deltaT;
            if (sqrt(pow(local_TA[i][0],2) + pow(local_TA[i][1],2))  > radius){
                local_TA_velocity[i][0] = -local_TA_velocity[i][0];
                local_TA_velocity[i][1] = -local_TA_velocity[i][1];

            }
            local_TA[i][0] += local_TA_velocity[i][0]*deltaT + accel_x*deltaT*deltaT/2; 
            local_TA[i][1] += local_TA_velocity[i][1]*deltaT + accel_y*deltaT*deltaT/2;

            TA_file << local_TA[i][0] << ", " << local_TA[i][1];
            if(i!=nmbr_TA-1){
                TA_file << ", ";
            }
        }
        TA_file << "\n";

        // Update NA positions
        for (int i=0; i<nmbr_NA;i++){
            local_NA[i][0] += local_NA_directions[i][0];
            local_NA[i][1] += local_NA_directions[i][1];
            NA_file << local_NA[i][0] << ", " << local_NA[i][1];
            if(i!=nmbr_NA-1){
                NA_file << ", ";
                }
        }
        NA_file << "\n";

        // Update referenced variables
        *NA_ptr = local_NA;
        *TA_ptr = local_TA;

        uint64_t ms_l = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        // Sleep to make 0.1 second each update
        std::this_thread::sleep_for(std::chrono::milliseconds(int(1000*deltaT - (ms_l-ms_i))) ); 
    }
    NA_file.close();
    TA_file.close();
    *finished_ptr = true;
}


// Function that get direction from dual variables and dC/dx

void get_directions(std::vector<std::vector<double>> agents, std::vector<std::vector<double>> NA, int nmb_agents, int nmb_NA, int d_connected,auto dual_vars,double max_delta, std::vector<std::vector<double>>(*NA_directions_ptr))
{
    int i,j;
    std::vector<std::vector<std::vector<double>>> deriv_rates(nmb_agents);
    get_dC(agents,NA,nmb_agents,nmb_NA,d_connected,&deriv_rates);

    // For each agents
    for (i=nmb_agents-nmb_NA;i<nmb_agents;i++){
        double direc_x = 0;
        double direc_y = 0;

        // Compute the direction of local ascent
        for(j=0;j<nmb_agents;j++){
            direc_x += deriv_rates[i][j][0]*dual_vars[i*nmb_agents + j] - deriv_rates[j][i][0]*dual_vars[j*nmb_agents + i];
            direc_y += deriv_rates[i][j][1]*dual_vars[i*nmb_agents + j] - deriv_rates[j][i][1]*dual_vars[j*nmb_agents + i];
        }
        direc_x = direc_x*0.1;
        direc_y = direc_y*0.1;

        // Top the norm of the direction to be the same than TA velocities
        double norm = sqrt(pow(direc_x,2) + pow(direc_y,2));
        if (norm >  max_delta/100){ 
            direc_x = direc_x / norm * max_delta/100;
            direc_y = direc_y / norm * max_delta/100;
        }
        (*NA_directions_ptr)[i-(nmb_agents-nmb_NA)][0] = direc_x;
        (*NA_directions_ptr)[i-(nmb_agents-nmb_NA)][1] = direc_y;
        
        //}
        //else {
        //(*NA_directions_ptr)[i-(nmb_agents-nmb_NA)][0] = direc_x/norm *4*max_delta/100;
        //(*NA_directions_ptr)[i-(nmb_agents-nmb_NA)][1] = direc_y/norm *4*max_delta/100;
        //}
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
    const double d_connected = 1;
    const int nmb_agents = I+A;
    const double radius = sqrt(nmb_agents / 3.1416) ;

    std::vector<std::vector<double>> TA(A);
    std::vector<std::vector<double>> NA(I);
    std::vector<std::vector<double>> agents(nmb_agents);
    bool finished;


    // Random set up
    std::uniform_real_distribution<double> rndm_rad(0,radius);
    std::uniform_real_distribution<double> rndm_ang(0,2*3.1416);
    std::default_random_engine re;

    // Define variables and parameters
    auto r_ijk = M->variable( new_array_ptr<int,1>({A,A+I,A+I}), Domain::greaterThan(0.)) ;
    auto a_k = M->variable( A, Domain::greaterThan(0.) ) ;
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
    M->objective(ObjectiveSense::Maximize, Expr::sum(a_k));
    // M->setSolverParam("optimizer", "dualSimplex");



    // LOOP OF EXPERIMENTS 
        
    std::vector<double> W(A);
    for (i=0; i < A; i++){
            W[i] = 0;
    }
    W[0] = 1;

    auto W_monty = monty::new_array_ptr<double>(W);
    weights_param->setValue(W_monty);

    int nmbr_experiments = std::atoi(argv[3]);
    for(int experiment=0;experiment<nmbr_experiments;experiment++){
        std::string path="../data_new/";
        path = path + std::to_string(A) + "TA"  +  std::to_string(I) +"NA/experiment_" + std::to_string(experiment);
        std::filesystem::create_directories(path);
        std::ofstream performance_file;
        performance_file.open (path + "/performance.txt",std::ios::out);

        // Initialize TA agents
        for(i=0;i<A;i++){
            double ang = rndm_ang(re);
            double rad = rndm_rad(re);
            TA[i] = std::vector<double>(2);
            TA[i][0]= rad*cos(ang);
            TA[i][1]= rad*sin(ang);
            agents[i] = std::vector<double>(2);
            agents[i][0]=TA[i][0];
            agents[i][1]=TA[i][1];

        }

        // Initialize NA agents
        for(i=0;i<I;i++){
            NA[i] = std::vector<double>(2);
            double ang = rndm_ang(re);
            double rad = rndm_rad(re);
            NA[i][0]= rad*cos(ang);
            NA[i][1]= rad*sin(ang);
            agents[A + i] = std::vector<double>(2);
            agents[A + i][0]=NA[i][0];
            agents[A + i][1]=NA[i][1];
        }

        // First solve for optimal initial positions with TA not moving

        const double lr_0 = 0.01;
        //const double decay = 0.99;
        double lr = lr_0;
        int max_iter = 100;
        int iter;

        for(iter=0;iter<max_iter;iter++){

            // Get value of C
            auto C = get_C(agents,nmb_agents,d_connected);
            C_param->setValue(C);

            // Solve problem
            M->solve();
            auto duals = *(c_links->dual());

            // Update NA positons
            //lr = lr*decay;
            update_positions(agents, NA, nmb_agents, I, d_connected,duals,lr,&NA);
            update_agents(TA,NA,A,I,&agents);

        }

        // Initialize on 0 NA_directions
        std::vector<std::vector<double>> NA_directions(I);
        for(i=0;i<I;i++){
            NA_directions[i] = std::vector<double>(2);
            NA_directions[i][0]=0;
            NA_directions[i][1]=0;
        }


        // Launch simulation thread that is going to update positions
        finished = false;
        double sim_duration = 20;
        double deltaT = 0.2;
        double max_vel = 90;
        std::thread th1(simulate, &TA,&NA,A,I,&NA_directions,sim_duration, deltaT,&finished,path);

        // Run until simultaion thread is over
        while(finished==false){

            // Get value of C
            auto C = get_C(agents,nmb_agents,d_connected);
            C_param->setValue(C);

            // Solve problem
            M->solve();
            performance_file <<  M->primalObjValue() << "\n";

            auto duals = *(c_links->dual());

            // Update NA positons
            get_directions(agents, NA, nmb_agents, I, d_connected, duals,max_vel*deltaT,&NA_directions);
            update_agents(TA,NA,A,I,&agents);
        }
        performance_file.close();
        th1.join();
        std::system("clear");
        std::cout<< experiment <<"/" << nmbr_experiments << "\n";    
    }
}


