#include <math.h>
#include "option.hpp"
#include "numa_helper.hpp"
#include "log.hpp"

#include "graph.hpp"
#include "solver.hpp"
//#include "partition.hpp"

struct PPR {
    double tp_;
    Graph* graph_;
    uint32_t *rooted_value;
    double *prb_of_steps;//Pre computing the probability of the step length VS the probability; 
    double tot_prb = 0.0;
    std::vector<vertex_id_t> start_vertices;
    int type = 0;
//    sfmt_t sfmt_;

    PPR() {}

    PPR(double tp, Graph* graph) : tp_(tp), graph_(graph) {
        rooted_value = new uint32_t[graph_->v_num]();
        prb_of_steps = new double[128]();
        for(int si = 0; si < 128; si++)
        {
            prb_of_steps[si] = pow((1-tp_), si) * tp_;
        }

        for(int si = 0; si < 127; si++)
        {
            prb_of_steps[si + 1] += prb_of_steps[si];
        }

        //printf("satrt get max degree vertex\n");
        vertex_id_t start_vertex = 0;
        vertex_id_t start_vertex_degree = graph_->adjlists[0][start_vertex].degree;
        for(vertex_id_t v_i = 0; v_i < graph_->v_num; v_i++)
        {
            vertex_id_t v_i_degree = graph_->adjlists[0][v_i].degree;
            if(v_i_degree > start_vertex_degree)
            {
                start_vertex = v_i;
            }
        }

        /*default_rand_t *rd = new default_rand_t[1];
        vertex_id_t start_vertex = rd->gen(graph->v_num);
        partition_id_t vertex_pid = graph->vertex_partition_id_fuc(start_vertex);
        while(start_vertex >= graph->partition_end[vertex_pid])
        {
            start_vertex = rd->gen(graph->v_num);
        }*/
        start_vertex = 0;
        printf("Root vertex %u\n", start_vertex);
        start_vertices.push_back(start_vertex);

        tot_prb = prb_of_steps[127];
    }

    PPR(const PPR& ppr) : PPR(ppr.tp_, ppr.graph_) {}

    inline vertex_id_t walking_step(default_rand_t* rd)
    {
        vertex_id_t stepid = 1;
        double genfloat;
        while(true)
        {
            genfloat = rd->gen_float(1.0);
            if(genfloat > tot_prb)
            {
                stepid += 128;
            }
            else
            {
                int si;
                for(si = 0; si < 128; si++)
                {
                    if(prb_of_steps[si] >= genfloat)
                    {
                        stepid += si;
                        break;
                    }
                }
                break;
            }
        }
        return stepid;
    }

    inline vertex_id_t walker_init(walker_id_t walker_id, default_rand_t* rd)
    {
        if(start_vertices.size() == 0)
        {
            return rd->gen(graph_->v_num);
        }
        else
        {
            if(start_vertices.size() == 1)
            {
                return start_vertices[0];
            }
            else
            {
                return start_vertices[rd->gen(start_vertices.size())];
            }
        }
    }

    inline void walker_pre_load(walker_t &mywalker)
    {
    }

    inline void update(walker_t &mywalker) {
        rooted_value[mywalker.vid]++;
        //return true;
    }
};


int main(int argc, char** argv)
{
    init_glog(argv, google::INFO);

    RWPTWalkOptionHelper opt;
    opt.parse(argc, argv);

    init_concurrency(opt.mtcfg);

    Graph graph(opt.mtcfg);
    graph.load(opt.graph_path.c_str(), opt.graph_format, false, opt.degree_bod);
    //make_graph(opt.graph_path.c_str(), opt.graph_format, true, opt.get_walker_num_func(), opt.walk_len, opt.mtcfg, opt.mem_quota, false, graph);
    PPR myapp(opt.start_prb, &graph);

    FMobSolver solver(&graph, opt.mtcfg);
    walk(&solver, opt.get_walker_num(graph.v_num), opt.walk_len, opt.mem_quota, myapp, opt.walker_size);
    printf("ppr value %u\n", myapp.rooted_value[0]);
    return 0;
}
