#include <math.h>
#include "option.hpp"
#include "numa_helper.hpp"
#include "log.hpp"

#include "graph.hpp"
#include "solver.hpp"
//#include "partition.hpp"

struct deepwalk {
    uint32_t tp_;
    Graph* graph_;
    uint32_t *rooted_value;
    int type = 0;
//    sfmt_t sfmt_;

    deepwalk() {}

    deepwalk(double tp, Graph* graph) : tp_(tp), graph_(graph) {
        rooted_value = new uint32_t[graph_->v_num]();
    }

    deepwalk(const deepwalk& mydeepwalk) : deepwalk(mydeepwalk.tp_, mydeepwalk.graph_) {}

    inline vertex_id_t walking_step(default_rand_t* rd)
    {
        return tp_ - 1;
    }

    inline void walker_pre_load(walker_t &mywalker)
    {
    }

    inline vertex_id_t walker_init(walker_id_t walker_id, default_rand_t* rd)
    {
        return rd->gen(graph_->v_num);
    }

    inline void update(walker_t &mywalker) {
    }
};


int main(int argc, char** argv)
{
    init_glog(argv, google::INFO);

    WalkOptionParser opt;
    opt.parse(argc, argv);

    init_concurrency(opt.mtcfg);

    Graph graph(opt.mtcfg);
    graph.load(opt.graph_path.c_str(), opt.graph_format, false);
    //make_graph(opt.graph_path.c_str(), opt.graph_format, true, opt.get_walker_num_func(), opt.walk_len, opt.mtcfg, opt.mem_quota, false, graph);

    deepwalk myapp(opt.walk_len, &graph);

    FMobSolver solver(&graph, opt.mtcfg);
    walk(&solver, opt.get_walker_num(graph.v_num), opt.walk_len, opt.mem_quota, myapp);
    printf("ppr value %u\n", myapp.rooted_value[0]);
    return 0;
}
