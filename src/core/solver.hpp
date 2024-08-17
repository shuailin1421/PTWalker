#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <sched.h>
#include <assert.h>
#include <mutex>

#include <iomanip>
#include <map>
#include <sys/types.h>
#include <unistd.h>

#include <omp.h>

#include "constants.hpp"
#include "timer.hpp"
#include "random.hpp"
#include "log.hpp"
#include "memory.hpp"
#include "graph.hpp"
//#include "walker.hpp"
//#include "message.hpp"
#include "sampler.hpp"
//#include "walk.hpp"
#include "compile_helper.hpp"
#include "profiler.hpp"
//#include "partition.hpp"
#include "perf_helper.hpp"

struct walker_t
{
    walker_id_t wid;
    vertex_id_t vid;
    vertex_id_t sid;
    //uint64_t w_data;
};

struct used_pool_units
{
    //std::vector<walker_t> walks_array;
    //walker_t* walks_array;
    uint16_t walks_start;
    uint16_t walks_end;
    uint32_t walks_size;
    walker_id_t rpoolid;
};

struct walkers_pos
{
    uint16_t walks_start;
    uint16_t walks_end;
};

struct unused_pool_units
{
    uint32_t pool_start;
    uint32_t pool_end;
};

/**
 * FMobSolver manages the whole random walk processing.
 */
class FMobSolver{
    MultiThreadConfig mtcfg;
    Graph* graph;
    default_rand_t** rands;
    MemoryPool mpool;

    uint64_t rest_walker_num;
    uint64_t terminated_walker_num;
    uint64_t max_epoch_walker_num;
    double total_walk_time;
    unsigned walk_len;
    double *message_time;
    walker_t ***walkers;
    walkers_pos **walker_pos;
    walker_id_t **unused_pool;
    unused_pool_units* unused_pool_pos;
    walker_id_t ***used_pool;
    used_pool_units **used_pool_pos;
    walker_id_t walks_per_container;
    walker_id_t *walks_units_size;
    walker_id_t walks_units_max_size;
    std::mutex *unused_pool_add_lock;
    std::mutex *unused_pool_remove_lock;
    std::mutex **used_pool_lock;
    std::mutex *partition_lock;

    bool is_node2vec;

    //MessageManager msgm;
    SamplerManager sm;
    //WalkManager wm;
    //WalkerManager wkrm;

    std::vector<vertex_id_t*> walks;

    bool is_hdv_thread (int t_id) {
        return (int)t_id < (mtcfg.thread_num + 1) / 2;
    }

public:
    SampleProfiler profiler;
    uint64_t* step_thread;
    //walker_id_t **computing_walkers;
    //walker_id_t **moved_walkers;

    uint64_t get_epoch_walker_num()
    {
        return(std::min(max_epoch_walker_num, rest_walker_num));
    }

    int get_thread_num()
    {
        return(mtcfg.thread_num);
    }

    FMobSolver(Graph* _graph, MultiThreadConfig _mtcfg) : mtcfg (_mtcfg), mpool(_mtcfg),  sm(_mtcfg), profiler(_graph->partition_num) {
        graph = _graph;
        is_node2vec = false;
        rands = nullptr;
    }

    ~FMobSolver() {
        if (rands != nullptr) {
            delete []rands;
        }

        /*if(walks_threads != nullptr)
        {
            //std::vector<std::vector<walkers_units>>().swap(walks_threads);
            for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
            {
                delete[] walks_threads[w_i];
            }
            delete []walks_threads;
        }

        if(walks_container != nullptr)
        {
            for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
            {
                for(partition_id_t p_i = 0; p_i < graph->partition_num; p_i++)
                {
                    delete[] walks_container[w_i][p_i];
                }
                delete[] walks_container[w_i];
            }
            delete[] walks_container;
        }*/
    }

    // Set node2vec, but don't prepare or initialize related data structure now.
    void set_node2vec(real_t _p, real_t _q) {
        is_node2vec = true;
        //wm.set_node2vec(_p, _q);
    }

    std::string name() {
        return std::string("FlashMob solver");
    }

    template<typename Alg>
    void myapp_update(partition_id_t p_i, Alg& myapp)
    {
        Timer message_timer;
        walker_id_t poolid;
        uint16_t walker_start;
        uint16_t walker_end;
        for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
        {
            auto used_pool_start = used_pool_pos[w_i][p_i].walks_start;
            auto used_pool_end = used_pool_pos[w_i][p_i].walks_end;

            for(walker_id_t wp_i = used_pool_start; wp_i < used_pool_end; wp_i++)
            {
                //used_pool_lock[w_i][p_i].lock();
                poolid = used_pool[w_i][p_i][wp_i];
                //used_pool_lock[w_i][p_i].unlock();
                walker_start = walker_pos[w_i][poolid].walks_start;
                walker_end = walker_pos[w_i][poolid].walks_end;

                for(uint16_t wk_i = walker_start; wk_i < walker_end; wk_i++)
                {
                    myapp.update(walkers[w_i][poolid][wk_i]);
                }
            }
        }
        message_time[0*8] += message_timer.duration();
    }

    template<typename sampler_t, typename Alg>
    walker_id_t walk_message(sampler_t *sampler, int w_i, partition_id_t p_i, Alg& myapp, bool sync = false) {
        Timer message_timer;
        double update_time;
        double tmp_message_time;
        int worker_id = omp_get_thread_num();
        auto *rd = this->rands[worker_id];
        auto used_pool_start = used_pool_pos[w_i][p_i].walks_start;
        auto used_pool_end = used_pool_pos[w_i][p_i].walks_end;
        //used_pool_pos[w_i][p_i].walks_start = used_pool_pos[w_i][p_i].walks_end - 1;
        partition_id_t tmp_partition;
        vertex_id_t vertex_old;
        vertex_id_t vertex_new;
        walker_id_t finished_walker = 0;
        uint16_t last_walker_start_recore;
        uint16_t last_walker_end_recore;
        walker_id_t poolid;
        uint16_t walker_start;
        uint16_t walker_end;
        walker_id_t stepid;
        uint16_t left_walkers;
        uint16_t walker_endmin1;
        uint16_t walker_pre_start;
        //if(worker_id == 0)
        //printf("%d %d %u start compute walkers\n", worker_id, w_i, p_i);
        update_time = message_timer.duration();
        if(sync == false)
        {
            if(myapp.type == 0)
            {
                for(walker_id_t wp_i = used_pool_start; wp_i < used_pool_end; wp_i++)
                {
                   //used_pool_lock[w_i][p_i].lock();
                    poolid = used_pool[w_i][p_i][wp_i];
                    //used_pool_lock[w_i][p_i].unlock();
                    walker_start = walker_pos[w_i][poolid].walks_start;
                    walker_end = walker_pos[w_i][poolid].walks_end;
                    //walker_pos[w_i][poolid].walks_start = walker_pos[w_i][poolid].walks_end;
                    last_walker_start_recore = walker_start;
                    last_walker_end_recore = walker_end;

                    for(uint16_t wk_i = walker_start; wk_i < walker_end; wk_i++)
                    {
                        /*if ((wk_i & 7) == 0) {
                            _mm_prefetch(&walkers[w_i][poolid][wk_i + 8], _MM_HINT_T2);
                            //_mm_clflushopt(&units[edge_idx]);
                        }*/
                        //vertex_old = walkers[w_i][poolid][wk_i].w_data & 0x000000000fffffff;
                        while(true)
                        {
                            myapp.update(walkers[w_i][poolid][wk_i]);
                            if(walkers[w_i][poolid][wk_i].sid > 0)
                            {
                                walkers[w_i][poolid][wk_i].vid = sampler->sample(walkers[w_i][poolid][wk_i].vid, rd);
                                tmp_partition = graph->vertex_partition_id_fuc(walkers[w_i][poolid][wk_i].vid);
                                walkers[w_i][poolid][wk_i].sid--;
                                if(tmp_partition != p_i)
                                {
                                    break;
                                }
                            }
                            else
                            {
                                finished_walker++;
                                //walkers[w_i][poolid][wk_i].w_data = 0xffffffffffffffff;
                                walkers[w_i][poolid][wk_i].vid = graph->v_num;
                                break;
                            }

                        }
                    }
                }
            }
            else
            {
                for(walker_id_t wp_i = used_pool_start; wp_i < used_pool_end; wp_i++)
                {
                   //used_pool_lock[w_i][p_i].lock();
                    poolid = used_pool[w_i][p_i][wp_i];
                    //used_pool_lock[w_i][p_i].unlock();
                    walker_start = walker_pos[w_i][poolid].walks_start;
                    walker_end = walker_pos[w_i][poolid].walks_end;
                    //walker_pos[w_i][poolid].walks_start = walker_pos[w_i][poolid].walks_end;
                    last_walker_start_recore = walker_start;
                    last_walker_end_recore = walker_end;
                    walker_endmin1 = walker_end - 1;

                    for(uint16_t wk_i = walker_start; wk_i < walker_end; wk_i++)
                    {
                        
                        while(true)
                        {
                            if(walkers[w_i][poolid][wk_i].sid > 0)
                            {
                                myapp.walker_pre_load(walkers[w_i][poolid][wk_i]);
                                walkers[w_i][poolid][wk_i].vid = sampler->sample(walkers[w_i][poolid][wk_i].vid, rd);
                                tmp_partition = graph->vertex_partition_id_fuc(walkers[w_i][poolid][wk_i].vid);
                                walkers[w_i][poolid][wk_i].sid--;
                                myapp.update(walkers[w_i][poolid][wk_i]);
                                if(tmp_partition != p_i)
                                {
                                    break;
                                }
                            }
                            else
                            {
                                finished_walker++;
                                //walkers[w_i][poolid][wk_i].w_data = 0xffffffffffffffff;
                                walkers[w_i][poolid][wk_i].vid = graph->v_num;
                                break;
                            }

                        }
                    }
                }
            }
        }
        else
        {
            for(walker_id_t wp_i = used_pool_start; wp_i < used_pool_end; wp_i++)
            {
                //used_pool_lock[w_i][p_i].lock();
                poolid = used_pool[w_i][p_i][wp_i];
                //used_pool_lock[w_i][p_i].unlock();
                walker_start = walker_pos[w_i][poolid].walks_start;
                walker_end = walker_pos[w_i][poolid].walks_end;
                //walker_pos[w_i][poolid].walks_start = walker_pos[w_i][poolid].walks_end;
                last_walker_start_recore = walker_start;
                last_walker_end_recore = walker_end;

                for(uint16_t wk_i = walker_start; wk_i < walker_end; wk_i++)
                {
                    /*if ((wk_i & 7) == 0) {
                        _mm_prefetch(&walkers[w_i][poolid][wk_i + 8], _MM_HINT_T2);
                        //_mm_clflushopt(&units[edge_idx]);
                    }*/
                    //vertex_old = walkers[w_i][poolid][wk_i].w_data & 0x000000000fffffff;
                    while(true)
                    {
                        //myapp.update(walkers[w_i][poolid][wk_i]);
                        if(walkers[w_i][poolid][wk_i].sid > 0)
                        {
                            walkers[w_i][poolid][wk_i].vid = sampler->sample(walkers[w_i][poolid][wk_i].vid, rd);
                            tmp_partition = graph->vertex_partition_id_fuc(walkers[w_i][poolid][wk_i].vid);
                            walkers[w_i][poolid][wk_i].sid--;
                            if(tmp_partition != p_i)
                            {
                                break;
                            }
                        }
                        else
                        {
                            finished_walker++;
                            //walkers[w_i][poolid][wk_i].w_data = 0xffffffffffffffff;
                            walkers[w_i][poolid][wk_i].vid = graph->v_num;
                            break;
                        }
                        break;
                    }
                }
            }
        }
        tmp_message_time = message_timer.duration();
        message_time[worker_id*8] += message_timer.duration() - update_time;

        uint16_t recore_pi = 0;
        walker_id_t mpoolid;
        walker_id_t new_poolid;
        uint16_t wk_i = 0;
        for(uint16_t up_i = used_pool_start; up_i < used_pool_end; up_i++)
        {
            
            recore_pi = up_i;
            //used_pool_lock[w_i][p_i].lock();
            poolid = used_pool[w_i][p_i][up_i];
            //used_pool_lock[w_i][p_i].unlock();
            walker_start = walker_pos[w_i][poolid].walks_start;
            walker_end = walker_pos[w_i][poolid].walks_end;
            if(up_i == used_pool_end - 1)
            {
                walker_start = last_walker_start_recore;
                walker_end = last_walker_end_recore;
            }
            for(wk_i = walker_start; wk_i < walker_end; wk_i++)
            {
                /*if ((wk_i & 7) == 0) {
                    _mm_prefetch(&walkers[w_i][poolid][wk_i + 8], _MM_HINT_T2);
                    //_mm_clflushopt(&units[edge_idx]);
                }*/
                if(walkers[w_i][poolid][wk_i].vid != graph->v_num)
                {
                    //vertex_old = walkers[w_i][poolid][wk_i].w_data & 0x000000000fffffff;
                    tmp_partition = graph->vertex_partition_id_fuc(walkers[w_i][poolid][wk_i].vid);
                    //move walks
                    //if(tmp_partition == p_i)
                    //{
                    //    printf("nor right\n");
                    //    continue;
                    //}
                    //else
                    {
                        mpoolid = used_pool_pos[worker_id][tmp_partition].rpoolid;
                        uint16_t &walkers_pos_end = walker_pos[worker_id][mpoolid].walks_end;
                        if(walker_pos[worker_id][mpoolid].walks_end < walks_per_container)
                        {
                            //walkers[worker_id][mpoolid][walkers_pos_end].w_data = walkers[w_i][poolid][wk_i].w_data;
                            walkers[worker_id][mpoolid][walkers_pos_end] = walkers[w_i][poolid][wk_i];
                            walkers_pos_end++;
                        }
                        else
                        {
                            if(unused_pool_pos[worker_id].pool_start == unused_pool_pos[worker_id].pool_end)
                            {
                                //printf("No pool left generate new walker array\n");
                                walker_id_t pool_size_left = walks_units_max_size - walks_units_size[worker_id];
                                walker_id_t generated_pool_size = walks_units_max_size / 16;
                                walker_id_t generated_pool_end = walks_units_size[worker_id] + generated_pool_size;
                                //unused_pool_pos[worker_id].pool_start = 0;
                                //unused_pool_pos[worker_id].pool_end = 0;
                                for(walker_id_t gpid = walks_units_size[worker_id]; gpid < generated_pool_end; gpid++)
                                {
                                    walkers[worker_id][gpid] = mpool.alloc_new<walker_t>(walks_per_container, 0);
                                }
                                unused_pool_add_lock[worker_id].lock();
                                for(walker_id_t gpid = walks_units_size[worker_id]; gpid < generated_pool_end; gpid++)
                                {
                                    unused_pool[worker_id][unused_pool_pos[worker_id].pool_end] = gpid;
                                    unused_pool_pos[worker_id].pool_end = (unused_pool_pos[worker_id].pool_end + 1) % walks_units_max_size;
                                }
                                unused_pool_add_lock[worker_id].unlock();
                                walks_units_size[worker_id] += generated_pool_size;
                                //printf("new generate %u arrays\n",generated_pool_size);
                            }
                            new_poolid = unused_pool[worker_id][unused_pool_pos[worker_id].pool_start];
                            unused_pool_pos[worker_id].pool_start = (unused_pool_pos[worker_id].pool_start + 1) % walks_units_max_size;
                            /*while(walker_pos[worker_id][new_poolid].walks_start != 0 || walker_pos[worker_id][new_poolid].walks_end != 0)
                            {
                                printf("error not zero %u %u\n", walker_pos[worker_id][new_poolid].walks_start, walker_pos[worker_id][new_poolid].walks_end);
                                new_poolid = unused_pool[worker_id][unused_pool_pos[worker_id].pool_start];
                                unused_pool_pos[worker_id].pool_start = (unused_pool_pos[worker_id].pool_start + 1) % walks_units_max_size;
                            }*/
                            walker_pos[worker_id][new_poolid].walks_start = 0;
                            walker_pos[worker_id][new_poolid].walks_end = 0;
                            used_pool_lock[worker_id][tmp_partition].lock();
                            if(used_pool_pos[worker_id][tmp_partition].walks_end >= used_pool_pos[worker_id][tmp_partition].walks_size)
                            {
                                //printf("walker size double\n");
                                walker_id_t* new_pool = new walker_id_t[used_pool_pos[worker_id][tmp_partition].walks_size * 2];
                                for(walker_id_t wp_i = 0; wp_i < used_pool_pos[worker_id][tmp_partition].walks_size; wp_i++)
                                {
                                    new_pool[wp_i] = used_pool[worker_id][tmp_partition][wp_i];
                                }
                                walker_id_t* tmp_array = used_pool[worker_id][tmp_partition];
                                used_pool[worker_id][tmp_partition] = new_pool;
                                delete[] tmp_array;
                                used_pool_pos[worker_id][tmp_partition].walks_size = used_pool_pos[worker_id][tmp_partition].walks_size * 2;
                            }
                            used_pool[worker_id][tmp_partition][used_pool_pos[worker_id][tmp_partition].walks_end] = new_poolid;
                            used_pool_pos[worker_id][tmp_partition].walks_end++;
                            used_pool_pos[worker_id][tmp_partition].rpoolid = new_poolid;
                            used_pool_lock[worker_id][tmp_partition].unlock();
                            //walkers[worker_id][new_poolid][walker_pos[worker_id][new_poolid].walks_end].w_data = walkers[w_i][poolid][wk_i].w_data;
                            walkers[worker_id][new_poolid][walker_pos[worker_id][new_poolid].walks_end] = walkers[w_i][poolid][wk_i];
                            walker_pos[worker_id][new_poolid].walks_end++;
                        }
                    }
                }
            }

            if(up_i < used_pool_end - 1)
            {
                //all pool should move to unused_pool with locking;
                //printf("release a pool %u %u\n", poolid, walker_end - walker_start);
                walker_pos[w_i][poolid].walks_start = 0;
                walker_pos[w_i][poolid].walks_end = 0;
                unused_pool_add_lock[w_i].lock();
                unused_pool[w_i][unused_pool_pos[w_i].pool_end] = poolid;
                unused_pool_pos[w_i].pool_end = (unused_pool_pos[w_i].pool_end + 1) % walks_units_max_size;
                unused_pool_add_lock[w_i].unlock();
                used_pool_pos[w_i][p_i].walks_start++;
            }
            else
            {
                walker_pos[w_i][poolid].walks_start = walker_end;
            }
        }
        //printf("%d -> walker compare %u %u\n",worker_id, last_computing_walker, (used_pool_end - 1) * walks_per_container + wk_i);
        if(recore_pi > 0)
        {
            used_pool_lock[w_i][p_i].lock();
            walker_id_t upcount = 0;
            for(walker_id_t upj = recore_pi; upj < used_pool_pos[w_i][p_i].walks_end; upj++)
            {
                used_pool[w_i][p_i][upcount] = used_pool[w_i][p_i][upj];
                upcount++;
            }
            used_pool_pos[w_i][p_i].walks_start = 0;
            used_pool_pos[w_i][p_i].walks_end = upcount;
            used_pool_lock[w_i][p_i].unlock();
        }

        message_time[worker_id*8 + 1] += message_timer.duration() - tmp_message_time;

        return finished_walker;
    }

    template<typename Alg>
    walker_id_t walk_message_dispatch(int w_i, partition_id_t p_i, Alg& myapp, bool sync = false) {
        auto *sampler = sm.samplers[p_i];
        if (sampler->sampler_class == ClassExclusiveBufferSampler) {
            //Timer buffer_time;
            return walk_message(static_cast<ExclusiveBufferSampler*>(sampler), w_i, p_i, myapp, sync);
        } else{
            return walk_message(static_cast<DirectSampler*>(sampler), w_i, p_i, myapp, sync);
        }
    }

    uint64_t estimate_epoch_walker(
    vertex_id_t vertex_num,
    edge_id_t edge_num,
    edge_id_t buffer_edge_num,
    uint64_t walker_num,
    int walk_len,
    int socket_num,
    uint64_t mem_quota,
    size_t other_size = 0
)
{
    #ifdef UNIT_TEST
    uint64_t temp_max_epoch_walker_num = std::min((uint64_t)vertex_num * 2u, walker_num);
    #else
    size_t graph_memory_size = sizeof(AdjList) * vertex_num * (size_t) socket_num + sizeof(AdjUnit) * edge_num;
    size_t buffer_memory_size = sizeof(vertex_id_t) * buffer_edge_num;
    size_t per_walker_cost = sizeof(vertex_id_t) * (
    // walk paths
    (walk_len * 2) \
    // messages + starting vertices
    + 2 + 1);
    // LOG(WARNING) << block_mid_str() << "Estimated memory size for graph data: " << size_string(graph_memory_size + buffer_memory_size + other_size);
    CHECK(mem_quota > graph_memory_size + buffer_memory_size + other_size) << "Assigned memory is too small to continue the computation";
    auto cal_max_active_walker_num = [&] (size_t memory_size) {
        uint64_t val = (memory_size - graph_memory_size - buffer_memory_size - other_size) / per_walker_cost;
        return val;
    };
    auto cal_epoch_num = [&] (size_t memory_size) {
        uint64_t temp_max_epoch_walker_num = std::min(cal_max_active_walker_num(memory_size), (uint64_t) walker_num);
        uint64_t epoch_num = (walker_num + temp_max_epoch_walker_num - 1) / temp_max_epoch_walker_num;
        return epoch_num;
    };

    auto cal_max_epoch_walker_num = [&] (size_t memory_size) {
        auto epoch_num = cal_epoch_num(memory_size);
        uint64_t temp_max_epoch_walker_num = (walker_num + epoch_num - 1) / epoch_num;
        return temp_max_epoch_walker_num;
    };

    uint64_t temp_max_epoch_walker_num = cal_max_epoch_walker_num(mem_quota);
    #endif
    temp_max_epoch_walker_num = std::min(temp_max_epoch_walker_num, (1ul << sizeof(walker_id_t) * 8ul) - 2ul);
    return temp_max_epoch_walker_num;
}

uint64_t walks_size_caculate(walker_id_t max_epoch_walkers)
    {
        
        walker_id_t max_thread_size = max_epoch_walkers;
        if(max_thread_size % walks_per_container != 0)
        {
            max_thread_size = (max_thread_size / walks_per_container + 1) * walks_per_container;
        }
        max_thread_size += (mtcfg.thread_num * graph->partition_num * 2) * walks_per_container;
        return max_thread_size;
    }

    void generate_walkers()
    {
        /*
        walker_t ***walkers;
        walkers_pos **walker_pos;
        walker_id_t **unused_pool;
        unused_pool_units* unused_pool_pos;
        walker_id_t ***used_pool;
        used_pool_units **used_pool_pos;
        walker_id_t walks_per_container;
        walker_id_t *walks_units_size;
        walker_id_t walks_units_max_size;
        */
        uint64_t max_walkers_num = walks_size_caculate(max_epoch_walker_num);
        walker_id_t max_pool_size = max_walkers_num / walks_per_container;
        walker_id_t thread_pool_size_init = max_pool_size / mtcfg.thread_num;
        if(max_pool_size % mtcfg.thread_num == 0)
        {
            thread_pool_size_init++;
        }
            
        walks_units_size = new walker_id_t[mtcfg.thread_num];
        for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
        {
            walks_units_size[w_i] = thread_pool_size_init + thread_pool_size_init / 8;
        }
        //printf("max pool size %u\n", thread_pool_size_init);
        walks_units_max_size = thread_pool_size_init * 2;
        walkers = new walker_t**[mtcfg.thread_num];
        #pragma omp parallel for
        for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
        {
            walkers[w_i] = new walker_t*[walks_units_max_size];
            for(walker_id_t pid = 0; pid < walks_units_size[w_i]; pid++)
            {
                walkers[w_i][pid] = mpool.alloc_new<walker_t>(walks_per_container, 0);
            }
        }

        MemoryCounter mcounter;
        mcounter.al_alloc<walkers_pos*>(mtcfg.thread_num);
        mcounter.al_alloc<walker_id_t*>(mtcfg.thread_num);
        mcounter.al_alloc<used_pool_units*>(mtcfg.thread_num);
        mcounter.al_alloc<unused_pool_units>(mtcfg.thread_num);

        for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
        {
            mcounter.al_alloc<walkers_pos>(walks_units_max_size);
            mcounter.al_alloc<walker_id_t>(walks_units_max_size);
            mcounter.al_alloc<used_pool_units>(graph->partition_num);
        }
        mcounter.align();
        Memory* m = mpool.get_memory(&mcounter, 0);
        walker_pos = m->al_alloc_new<walkers_pos*>(mtcfg.thread_num);
        unused_pool = m->al_alloc_new<walker_id_t*>(mtcfg.thread_num);
        used_pool_pos = m->al_alloc_new<used_pool_units*>(mtcfg.thread_num);
        unused_pool_pos = m->al_alloc_new<unused_pool_units>(mtcfg.thread_num);
        for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
        {
            walker_pos[w_i] = m->al_alloc_new<walkers_pos>(walks_units_max_size);
            unused_pool[w_i] = m->al_alloc_new<walker_id_t>(walks_units_max_size);
            used_pool_pos[w_i] = m->al_alloc_new<used_pool_units>(graph->partition_num);
        }

        #pragma omp parallel for
        for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
        {
            for(walker_id_t w_j = 0; w_j < walks_units_max_size; w_j++)
            {
                walker_pos[w_i][w_j].walks_start = 0;
                walker_pos[w_i][w_j].walks_end = 0;
            }
        }
        //initialize use pool walker_id_t ***used_pool;
        used_pool = new walker_id_t**[mtcfg.thread_num];
        walker_id_t used_pool_size = walks_units_max_size * 8 / graph->partition_num + 1;
        if(used_pool_size % 8 != 0)
        {
            used_pool_size = used_pool_size / 8 * 8 + 8;
        }
        #pragma omp parallel for
        for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
        {
            used_pool[w_i] = new walker_id_t*[graph->partition_num];
            for(partition_id_t p_i = 0; p_i < graph->partition_num; p_i++)
            {
                used_pool[w_i][p_i] = new walker_id_t[used_pool_size];
                used_pool[w_i][p_i][0] = p_i;
                used_pool_pos[w_i][p_i].walks_start = 0;
                used_pool_pos[w_i][p_i].walks_end = 1;
                used_pool_pos[w_i][p_i].walks_size = used_pool_size;
                used_pool_pos[w_i][p_i].rpoolid = used_pool[w_i][p_i][0];
            }
            walker_id_t unused_pool_size = graph->partition_num;
            walker_id_t pool_count = 0;
            for(walker_id_t p_i = unused_pool_size; p_i < walks_units_size[w_i]; p_i++)
            {
                unused_pool[w_i][pool_count] = p_i;
                pool_count++; 
            }
            unused_pool_pos[w_i].pool_start = 0;
            unused_pool_pos[w_i].pool_end = pool_count;
        }
    }

    void prepare(uint64_t _walker_num, int _walk_len, uint64_t mem_quota, uint16_t walker_size = 4196) {
        LOG(WARNING) << block_begin_str() << "Initialize Solver";
        Timer timer;
        step_thread = new uint64_t[mtcfg.thread_num * 8]();
        message_time = new double[mtcfg.thread_num * 8]();
        used_pool_lock = mpool.alloc_new<std::mutex*>(mtcfg.thread_num, 0);
        unused_pool_add_lock = mpool.alloc_new<std::mutex>(mtcfg.thread_num, 0);
        unused_pool_remove_lock = mpool.alloc_new<std::mutex>(mtcfg.thread_num, 0);
        partition_lock = mpool.alloc_new<std::mutex>(graph->partition_num, 0);


        #pragma omp parallel for
        for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
        {
            used_pool_lock[w_i] = mpool.alloc_new<std::mutex>(graph->partition_num, 0);
        }
        walks_per_container = walker_size;

        rands = new default_rand_t*[mtcfg.thread_num];
        for (int t_i = 0; t_i < mtcfg.thread_num; t_i++) {
            // Don't use StdRandNumGenerator here,
            // as it does not support numa-aware allocation.
            rands[t_i] = mpool.alloc_new<default_rand_t>(1, mtcfg.socket_id(t_i));
        }
        LOG(WARNING) << block_mid_str() << "RandNumGenerator: " << rands[0]->name();

        rest_walker_num = 0;
        terminated_walker_num = 0;
        total_walk_time = 0;
        walk_len = 0;
        max_epoch_walker_num = 0;

        rest_walker_num = _walker_num;
        terminated_walker_num = 0;
        walk_len = _walk_len;

        if (is_node2vec) {
            graph->prepare_neighbor_query();
        }

        edge_id_t buffer_edge_num = 0;
#pragma omp parallel for reduction (+: buffer_edge_num)
        for (partition_id_t p_i = 0; p_i < graph->partition_num; p_i++) {
            if (graph->partition_sampler_class[p_i] == ClassExclusiveBufferSampler) {
                for (vertex_id_t v_i = graph->partition_begin[p_i]; v_i < graph->partition_end[p_i]; v_i++) {
                    edge_id_t vertex_buff_size = _walker_num * walk_len * graph->adjlists[0][v_i].degree / graph->e_num;
                    if(vertex_buff_size >= 8)
                    {
                        buffer_edge_num += vertex_buff_size;
                    }
                    else
                    {
                        buffer_edge_num += 8;
                    }
                }
            }
        }
        size_t ht_size = is_node2vec ? graph->bf->size() : 0;
        uint64_t temp_max_epoch_walker_num = estimate_epoch_walker(graph->v_num, graph->e_num, buffer_edge_num, _walker_num, 5, mtcfg.socket_num, mem_quota, ht_size);
        std::stringstream epoch_walker_ss;
        int epoch_num = 0;
        for (uint64_t w_i = 0; w_i < _walker_num;) {
            uint64_t epoch_walker_num = std::min(temp_max_epoch_walker_num, _walker_num - w_i);
            epoch_walker_ss << " " << epoch_walker_num;
            w_i += epoch_walker_num;
            epoch_num ++;
        }
        #if PROFILE_IF_BRIEF
        LOG(INFO) << block_mid_str() << "Total walkers: " <<  _walker_num << ", max_epoch_walkers: " << temp_max_epoch_walker_num << ", total epochs: " << epoch_num;
        LOG(INFO) << block_mid_str() << "Epoch walkers: " << epoch_walker_ss.str();
        LOG(WARNING) << block_mid_str() << "Walker density: " << (double) temp_max_epoch_walker_num / graph->e_num;
        #endif
        //max_epoch_walker_num = temp_max_epoch_walker_num;
        max_epoch_walker_num = _walker_num;
        
        //printf("start to sm init\n");
        sm.init(graph, temp_max_epoch_walker_num, &profiler, _walker_num, walk_len, graph->e_num);
        //wm.init(graph, &sm, &msgm, rands, &profiler);
        //wkrm.init(temp_max_epoch_walker_num);
        //msgm.init(graph, &wkrm, &profiler, is_node2vec);
        //init_start_walks(temp_max_epoch_walker_num);

        LOG(WARNING) << block_end_str() << "Solver initialized in " << timer.duration() << " seconds";
    }

    template<typename Alg>
    void init_start_walks(walker_id_t epoch_walker_num, Alg myapp)
    {
        //printf("init_start_walks\n");
        #pragma omp parallel for 
        for(uint32_t w_i = 0; w_i < epoch_walker_num; w_i++)
        {
            int worker_id = omp_get_thread_num();
            //uint64_t start_vertex = rands[omp_get_thread_num()]->gen(graph->v_num);
            uint32_t start_vertex = myapp.walker_init(w_i, rands[worker_id]);
            partition_id_t vertex_pid = graph->vertex_partition_id_fuc(start_vertex);
            //vertex_id_t fail_count = 0;
            while(start_vertex >= graph->partition_end[vertex_pid])
            {
                //start_vertex = rands[worker_id]->gen(graph->v_num);
                //fail_count++;
                start_vertex = myapp.walker_init(w_i, rands[worker_id]);
                vertex_pid = graph->vertex_partition_id_fuc(start_vertex);
            }
            // using pool to init walkers

            auto poolid = used_pool_pos[worker_id][vertex_pid].rpoolid;
            auto &walkers_pos_end = walker_pos[worker_id][poolid].walks_end;
            if(walker_pos[worker_id][poolid].walks_end < walks_per_container)
            {
                //walkers[worker_id][poolid][walkers_pos_end].w_data = start_vertex + w_i * 0x0000000010000000;
                walkers[worker_id][poolid][walkers_pos_end].vid = start_vertex;
                walkers[worker_id][poolid][walkers_pos_end].sid = myapp.walking_step(rands[worker_id]);
                walkers[worker_id][poolid][walkers_pos_end].wid = w_i;
                walkers_pos_end++;
            }
            else
            {
                //require a new pool;
                auto new_poolid = unused_pool[worker_id][unused_pool_pos[worker_id].pool_start];
                unused_pool_pos[worker_id].pool_start = (unused_pool_pos[worker_id].pool_start + 1) % walks_units_max_size;
                walker_pos[worker_id][new_poolid].walks_start = 0;
                walker_pos[worker_id][new_poolid].walks_end = 0;
                //TBD
                if(used_pool_pos[worker_id][vertex_pid].walks_end >= used_pool_pos[worker_id][vertex_pid].walks_size)
                {
                    //printf("walk size double\n");
                    walker_id_t* new_pool = new walker_id_t[used_pool_pos[worker_id][vertex_pid].walks_size * 2];
                    walker_id_t pool_count = 0;
                    for(walker_id_t w_i = used_pool_pos[worker_id][vertex_pid].walks_start; w_i < used_pool_pos[worker_id][vertex_pid].walks_end; w_i++)
                    {
                        new_pool[pool_count] = used_pool[worker_id][vertex_pid][w_i];
                        pool_count++;
                    }
                    delete[] used_pool[worker_id][vertex_pid];
                    used_pool[worker_id][vertex_pid] = new_pool;
                    used_pool_pos[worker_id][vertex_pid].walks_start = 0;
                    used_pool_pos[worker_id][vertex_pid].walks_end = pool_count;
                    used_pool_pos[worker_id][vertex_pid].walks_size = used_pool_pos[worker_id][vertex_pid].walks_size * 2;

                }
                used_pool[worker_id][vertex_pid][used_pool_pos[worker_id][vertex_pid].walks_end] = new_poolid;
                used_pool_pos[worker_id][vertex_pid].walks_end++;
                used_pool_pos[worker_id][vertex_pid].rpoolid = new_poolid;
                //walkers[worker_id][new_poolid][walker_pos[worker_id][new_poolid].walks_end].w_data = start_vertex + w_i * 0x0000000010000000;
                walkers[worker_id][new_poolid][walker_pos[worker_id][new_poolid].walks_end].vid = start_vertex;
                walkers[worker_id][new_poolid][walker_pos[worker_id][new_poolid].walks_end].sid = myapp.walking_step(rands[worker_id]);
                walkers[worker_id][new_poolid][walker_pos[worker_id][new_poolid].walks_end].wid = w_i;
                walker_pos[worker_id][new_poolid].walks_end++;
            }
        }

        walker_id_t init_walkers = 0;
        for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
        {
            for(partition_id_t p_i = 0; p_i < graph->partition_num; p_i++)
            {
                auto used_pool_start = used_pool_pos[w_i][p_i].walks_start;
                auto used_pool_end = used_pool_pos[w_i][p_i].walks_end;
                for(walker_id_t pi = used_pool_start; pi < used_pool_end; pi++)
                {
                    auto poolid = used_pool[w_i][p_i][pi];
                    /*for(uint16_t walker_start = walker_pos[w_i][poolid].walks_start; walker_start < walker_pos[w_i][poolid].walks_end; walker_start++)
                    {
                        if(graph->vertex_partition_id_fuc(walkers[w_i][poolid][walker_start].w_data & 0x000000000fffffff) != p_i)
                        {
                            printf("error init walkers\n");
                        }
                    }*/
                    init_walkers += walker_pos[w_i][poolid].walks_end - walker_pos[w_i][poolid].walks_start;
                }
            }
        }

        //printf("init walks %u\n", init_walkers);
    }

    template<typename Alg>
    void walk(walker_id_t &epoch_walker_num, Alg& myapp)
    {
        Timer timer;
        //const walker_id_t _walker_num = epoch_walker_num;
        epoch_walker_num = std::min(max_epoch_walker_num, rest_walker_num);

        init_start_walks(epoch_walker_num, myapp);

        printf("init walks start %lf\n", timer.duration());

        //printf("partition_num %d,hub partition num %d %d\n", graph->partition_num, graph->partition_num_pre_subgraph, graph->partition_num_pre_subgraph_left);
        //printf("pre left %d, pre right %d, dir left %d, dir right %d\n", graph->partition_num_pre_subgraph_left, graph->partition_num_pre_subgraph, graph->partition_num_low_left, graph->partition_num);

        walker_id_t finished_walkers = 0;
        walker_id_t* finished_walker_threads = new walker_id_t[mtcfg.thread_num * 8]();
        int iter = 0;
        //double tmp_time;
        //double thread_t = 0;
        //edge_id_t* step_thread = new edge_id_t[mtcfg.thread_num]();
        partition_id_t processing_subrgaph_iter = graph->partition_num + graph->partition_num_pre_subgraph;
        partition_id_t *subgraph_map = new partition_id_t[processing_subrgaph_iter]();
        partition_id_t map_pos = 0;
        for(partition_id_t p_i = 0; p_i < graph->partition_num_pre_subgraph_left; p_i++)
        {
            subgraph_map[map_pos] = p_i;
            map_pos++;
        }

        for(partition_id_t p_i = graph->partition_num_pre_subgraph; p_i < graph->partition_num_low_left; p_i++)
        {
            subgraph_map[map_pos] = p_i;
            map_pos++;
        }

        for(partition_id_t p_i = 0; p_i < graph->partition_num_pre_subgraph_left; p_i++)
        {
            subgraph_map[map_pos] = p_i;
            map_pos++;
        }

        for(partition_id_t p_i = graph->partition_num_pre_subgraph_left; p_i < graph->partition_num_pre_subgraph; p_i++)
        {
            subgraph_map[map_pos] = p_i;
            map_pos++;
        }

        for(partition_id_t p_i = graph->partition_num_low_left; p_i < graph->partition_num; p_i++)
        {
            subgraph_map[map_pos] = p_i;
            map_pos++;
        }

        for(partition_id_t p_i = graph->partition_num_pre_subgraph_left; p_i < graph->partition_num_pre_subgraph; p_i++)
        {
            subgraph_map[map_pos] = p_i;
            map_pos++;
        }

        assert(map_pos == processing_subrgaph_iter);
        std::vector<std::vector<partition_id_t> > huge_subgraph;
        for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
        {
            huge_subgraph.push_back(std::vector<partition_id_t>() );
        }
        int number_of_huge_vertex = 0;

        //Timer update_time;

        while(finished_walkers < epoch_walker_num)
        {
            uint64_t tot_steps = 0;
            for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
            {
                tot_steps += step_thread[w_i * 8];
            }
            printf("***********");
            printf("iter %d, finished_walkers %u totstep %lu time %lf\n", iter, finished_walkers, tot_steps, timer.duration());
            iter++;
            #pragma omp parallel for
            for(partition_id_t p_i = 0; p_i < graph->partition_num_pre_subgraph; p_i++)
            {
                //()sm.samplers[p_i]
                static_cast<ExclusiveBufferSampler*>(sm.samplers[p_i])->_walker_num = epoch_walker_num - finished_walkers;
            }
            //double* p_time = new double[2]();
            partition_id_t processing_iter = 0;
            walker_id_t max_processed_walker_per_subgraph = epoch_walker_num / mtcfg.thread_num / mtcfg.thread_num * 8 / 10;


            //double thread_timeqqq = 0;

            
            #pragma omp parallel
            {
                partition_id_t p_i_pos;
                auto worker_id = omp_get_thread_num();
                while((p_i_pos =  __sync_fetch_and_add(&processing_iter, 1)) < processing_subrgaph_iter)
                {
                    partition_id_t p_i = subgraph_map[p_i_pos];
                    //tmp_time = timer.duration();
                    Timer thread_time;
                    walker_id_t thread_Walker = 0;
                    //message_time[worker_id * 8] = 0;
                    walker_id_t walker_start;
                    walker_id_t walker_end;
                    walker_id_t walker_size1;
                    partition_lock[p_i].lock();
                    for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
                    {
                        walker_id_t walker_size = 0;
                        for(walker_id_t unpi = used_pool_pos[w_i][p_i].walks_start; unpi < used_pool_pos[w_i][p_i].walks_end; unpi++)
                        {
                            walker_id_t upoolid = used_pool[w_i][p_i][unpi];
                            walker_size += walker_pos[w_i][upoolid].walks_end - walker_pos[w_i][upoolid].walks_start;
                        }
                        /*if(walker_size >= max_processed_walker_per_subgraph && p_i >= graph->partition_num_pre_subgraph)
                        {
                            huge_subgraph[worker_id].push_back(p_i);
                            number_of_huge_vertex = 1;
                            break;
                        }*/
                        if(walker_size != 0)
                        {
                            finished_walker_threads[worker_id * 8] += walk_message_dispatch(w_i, p_i, myapp);
                            //printf("  --> worker %d, subgraph %u, time %lf ,walker num %u, message time %lf, %lf ns\n", worker_id, p_i, thread_time.duration(), thread_Walker, message_time[worker_id], (thread_time.duration() * 1000000000) / thread_Walker);
                        }
                        thread_Walker += walker_size;
                    }
                    partition_lock[p_i].unlock();
                    /*if(thread_Walker != 0)
                    {
                        printf("  --> worker %d, subgraph %u, time %lf ,walker num %u, message time %lf, %lf ns, %u %u %u\n", worker_id, p_i, thread_time.duration(), thread_Walker, message_time[worker_id], (thread_time.duration() * 1000000000) / thread_Walker, walker_start, walker_end, walker_size1);
                    }*/
                }
            }

            if(number_of_huge_vertex != 0)
            {
                for(int s_i = 0; s_i < huge_subgraph.size(); s_i++)
                {
                    for(int s_j = 0; s_j < huge_subgraph[s_i].size(); s_j++)
                    {
                        partition_id_t tmp_pt = huge_subgraph[s_i][s_j];
                        //printf("hub vertex graph %u\n", tmp_pt);
                        myapp_update(tmp_pt, myapp);
                        #pragma omp parallel for
                        for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
                        {
                            auto worker_id = omp_get_thread_num();
                            finished_walker_threads[worker_id * 8] += walk_message_dispatch(w_i, tmp_pt, myapp, true);
                        }
                    }
                    if(huge_subgraph[s_i].size() != 0)
                    {
                        huge_subgraph[s_i].clear();
                    }
                }
                number_of_huge_vertex = 0;
            }

            finished_walkers = 0;
            for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
            {
                finished_walkers += finished_walker_threads[w_i * 8];
            }
        }
        //printf("update time %lf\n", update_time.duration());
        double update_walker_time = 0;
        double send_walker_time = 0;
        for(int w_i = 0; w_i < mtcfg.thread_num; w_i++)
        {
            update_walker_time += message_time[w_i*8];
            send_walker_time += message_time[w_i*8 + 1];
        }
        printf("update %lf message %lf\n", update_walker_time, send_walker_time);

        rest_walker_num -= epoch_walker_num;
    }

    /*void walk_info() {
        uint64_t terminated_walk_step = (uint64_t) walk_len * terminated_walker_num;

        LOG(WARNING) << "time: " << total_walk_time << " s" \
            << ", step: " << number_string(terminated_walk_step) \
            << ", throughput: " << number_string(terminated_walk_step / total_walk_time) << "/s" \
            << ", speed: " << get_step_cost(total_walk_time, terminated_walk_step, mtcfg.thread_num) << " ns";

        std::cout << "time: " << total_walk_time << " s" \
            << ", step: " << number_string(terminated_walk_step) \
            << ", throughput: " << number_string(terminated_walk_step / total_walk_time) << "/s" \
            << ", speed: " << get_step_cost(total_walk_time, terminated_walk_step, mtcfg.thread_num) << " ns" << std::endl;
    }*/

    bool has_next_walk() {
        return (rest_walker_num != 0);
    }
};

template<typename Alg>
void walk(FMobSolver * solver, uint64_t walker_num, int walk_len, uint64_t mem_quota, Alg& myapp, uint16_t walker_size = 4196) {
    LOG(WARNING) << split_line_string() << "walker size " << walker_size;
    //printf("start to init\n");
    solver->prepare(walker_num, walk_len, mem_quota, walker_size);
    solver->generate_walkers();
    Timer tottimer;

    printf("start to walk %lf\n", tottimer.duration());

    uint64_t terminated_walker_num = 0;
    while (solver->has_next_walk()) {
        walker_id_t epoch_walker_num;
        epoch_walker_num = solver->get_epoch_walker_num();
        solver->walk(epoch_walker_num, myapp);
        terminated_walker_num += epoch_walker_num;
    }

    uint64_t totstep = 0;
    for(int w_i = 0; w_i < solver->get_thread_num(); w_i++)
    {
        totstep += solver->step_thread[w_i];
    }

    printf("total time (including init start walker) %lf, finished step %lu\n", tottimer.duration(), totstep);
}
