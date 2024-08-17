#pragma once

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <vector>
#include <unordered_map>
#include <cstddef>
#include <memory>
#include <algorithm>
#include <fstream>
#include <string>
#include <iostream>

#include <omp.h>

#include "type.hpp"
#include "util.hpp"
#include "timer.hpp"
#include "log.hpp"
#include "io.hpp"
#include "constants.hpp"
#include "random.hpp"
#include "memory.hpp"
#include "hash.hpp"

struct AdjUnit {
    vertex_id_t neighbor;
};

struct AdjList {
    vertex_id_t degree;
    AdjUnit *begin;
} __attribute__((packed));

/**
 * Brief description of a group used in Graph class.
 *
 * The partitions in the group starts from partition_offset.
 * Each partition has (1 << partition_bits) vertices.
 *
 */
struct GroupHeader {
    vertex_id_t partition_bits;
    vertex_id_t partition_offset;
};

/**
 * Description of a group to help Graph class partition the graph.
 *
 * The vertices in the group are [vertex_begin, vertex_end).
 * There are partition_num partitions in the group, each has
 * (1 << partition_bits) vertices. The estimated total time for
 * all the walkers on vertices within this group to take one 1 step
 * is total_time, the average of which is step_time. Partition_level
 * larger than 0 indicates that the partitions in this group will be
 * further partitioned into sub-partitions. Currently even it's added
 * to the model, on all evaluated graphs only the single level partitioning
 * is selected by DP algorithm. Additional level shuffling needs to be
 * added here.
 *
 */
struct GroupHint {
    vertex_id_t vertex_begin;
    vertex_id_t vertex_end;
    vertex_id_t partition_bits;
    vertex_id_t partition_num;
    double total_time;
    double step_time;
    int partition_level;
};

/**
 * Description of the groups and partitions to help Graph class partition the graph.
 *
 * Each group has (1<< group_bits) vertices, and there are group_num groups in total.
 * The partition_sampler_class has #partitions elements, giving the sampler type
 * suggestions for each partition.
 *
 */
struct GraphHint {
    vertex_id_t group_bits;
    vertex_id_t group_num;
    std::vector<GroupHint> group_hints;
    std::vector<SamplerClass> partition_sampler_class;
};

/**
 * Graph class loads graph from files, and manages all the vertices, edges, partitions, and groups.
 *
 * Partitions are evenly distributed to all availabel NUMA node.
 * Vertices are sorted by degree, except for the vertices of the first few
 * partitions, which are evenly shuffled for load-balance.
 *
 */
class Graph
{
    struct VertexSortUnit {
        vertex_id_t vertex;
        vertex_id_t degree;
    };
protected:
    MultiThreadConfig mtcfg;
    MemoryPool mpool;
public:

    std::unique_ptr<AdjList*[]> adjlists; // AdjList [sockets][vertices]
    std::unique_ptr<AdjUnit*[]> edges; // AdjUnit [sockets][vertices / sockets]
    vertex_id_t v_num;
    edge_id_t e_num;
    bool as_undirected;

    vertex_id_t *id2name; // vertex_id_t [vertices] (interleaved)

    std::vector<vertex_id_t> partition_begin;
    std::vector<vertex_id_t> partition_end;
    std::vector<SamplerClass> partition_sampler_class;
    std::vector<int> partition_socket;
    std::vector<edge_id_t> partition_edge_num;
    std::unique_ptr<int*[]> socket_partitions;
    std::unique_ptr<int[]> socket_partition_nums;

    vertex_id_t v_num_old;

    vertex_id_t *reorderid;
    partition_id_t partition_num;
    partition_id_t partition_num_pre_subgraph;
    partition_id_t partition_num_pre_subgraph_left;
    partition_id_t partition_num_low_left;
    vertex_id_t fat_subgraph_pos;
    vertex_id_t vertices_per_dir_subgraph;
    vertex_id_t vertices_per_pre_subgraph;
    edge_id_t edge_per_subgraph;


    partition_id_t* vertex_partition_id;
    std::vector<vertex_id_t> hub_vertex_group;

    // For node2vec
    std::unique_ptr<BloomFilter> bf;

    // Temporary variables, which will be cleared after making graph.
    std::vector<vertex_id_t> degrees;
    std::vector<Edge> raw_edges;
    std::vector<vertex_id_t> name2id;
    //std::vector<VertexSortUnit> vertex_units;
    //std::vector<edge_id_t> degree_prefix_sum;

    Graph(MultiThreadConfig _mtcfg) : mpool (_mtcfg) {
        mtcfg = _mtcfg;
        id2name = nullptr;
    }

    ~Graph() {
    }

    partition_id_t vertex_partition_id_fuc(vertex_id_t start_vertex)
    {
        if(start_vertex < fat_subgraph_pos)
        {
            return(start_vertex / vertices_per_pre_subgraph);
        }
        else
        {
            return(partition_num_pre_subgraph + (start_vertex - fat_subgraph_pos) / vertices_per_dir_subgraph);
        }
    }

    //Currently we just do hash partition
    //then try fennel based partition
    void max_cut_partition(partition_id_t* partition_id_of_maxcut, vertex_id_t* partition_vertex_num, AdjList* raw_adjlists, AdjUnit* raw_adjunits)
    {
        double* partition_score = new double[2]();
        vertex_id_t* connected_edges = new vertex_id_t[2]();

        for(vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {
            vertex_id_t tmp_vertex = v_i;
            if(partition_id_of_maxcut[tmp_vertex] == 2)
            {
                connected_edges[0] = 0;
                connected_edges[1] = 0;
                for(vertex_id_t n_i = 0; n_i < raw_adjlists[tmp_vertex].degree; n_i++)
                {
                    AdjUnit* begin = raw_adjlists[tmp_vertex].begin + n_i;
                    auto vertex_neighbor = begin->neighbor;
                    partition_id_t vertex_partition_id = partition_id_of_maxcut[vertex_neighbor];
                    if(vertex_partition_id != 3)
                    {
                        connected_edges[vertex_partition_id]++;
                    }
                }
                partition_score[0] = -double(connected_edges[0]);// - 1.5 * 1.5 * pow(double(partition_vertex_num[0]), 1.5);
                partition_score[1] = -double(connected_edges[1]);// - 1.5 * 1.5 * pow(double(partition_vertex_num[1]), 1.5);

                if(partition_score[0] > partition_score[1])
                {
                    partition_id_of_maxcut[v_i] = 0;
                    partition_vertex_num[0]++;
                }
                else
                {
                    partition_id_of_maxcut[v_i] = 1;
                    partition_vertex_num[1]++;
                }
            }
        }

    }

   

    void dp_partition(std::vector<VertexSortUnit> &PDdegree, std::vector<std::vector<vertex_id_t> > &vertex_subgroup, std::vector<edge_id_t> &vertex_subgroup_edges, uint64_t subgraph_size, double avg_degree, vertex_id_t avg_vertex)
    {
        //std::sort(PDdegree.begin(), PDdegree.end(), [](const VertexSortUnit& a, const VertexSortUnit& b){return a.degree > b.degree;});
        std::random_shuffle(PDdegree.begin(), PDdegree.end());
        vertex_id_t partitioned_vertex = 0;
        vertex_id_t front = 0;
        vertex_id_t end = PDdegree.size() - 1;
        bool terminated = false;
        while(!terminated)
        {
            uint64_t cur_size = 0;
            double subgraph_vertex = 0;
            double subgraph_edge = 0;
            double cur_degree = 0;
            vertex_id_t avg_vertex_tmp = avg_vertex * 11 / 10;
            std::vector<vertex_id_t> tmp_subgraph;
            while(cur_size < subgraph_size && front <= end) // && subgraph_vertex < avg_vertex_tmp)
            {
                tmp_subgraph.push_back(PDdegree[front].vertex);
                subgraph_vertex++;
                subgraph_edge += PDdegree[front].degree;
                cur_degree = subgraph_edge / subgraph_vertex;
                cur_size += 8 + (PDdegree[front].degree) * 4;
                front++;
                partitioned_vertex++;
                /*
                if(cur_degree <= avg_degree)
                {
                    tmp_subgraph.push_back(PDdegree[front].vertex);
                    subgraph_vertex++;
                    subgraph_edge += PDdegree[front].degree;
                    cur_degree = subgraph_edge / subgraph_vertex;
                    cur_size += 8 + (PDdegree[front].degree) * 4;
                    front++;
                    partitioned_vertex++;
                }
                else
                {
                    tmp_subgraph.push_back(PDdegree[end].vertex);
                    subgraph_vertex++;
                    subgraph_edge += PDdegree[end].degree;
                    cur_degree = subgraph_edge / subgraph_vertex;
                    cur_size += 8 + (PDdegree[end].degree) * 4;
                    end--;
                    partitioned_vertex++;
                }*/
            }
            vertex_subgroup.push_back(tmp_subgraph);
            vertex_subgroup_edges.push_back(edge_id_t(subgraph_edge));

            if(front > end)
            {
                terminated = true;
                if(partitioned_vertex != PDdegree.size())
                {
                    printf("It is Wrong!\n");
                }
            }
        }

    }

    void load(const char* path, GraphFormat graph_format, bool _as_undirected = true, vertex_id_t bodary = 32) {
        LOG(WARNING) << block_begin_str(1) << "Load graph" << "bodary " << bodary;
        Timer timer;
        as_undirected = _as_undirected;
        e_num = 0;
        v_num = 0;
        if (graph_format == BinaryGraphFormat) {
            read_binary_graph(path, raw_edges);
        } else {
            read_text_graph(path, raw_edges);
        }
        e_num = raw_edges.size();
        if (as_undirected) {
            //e_num = raw_edges.size() * 2;
            raw_edges.resize(e_num * 2);
        }
        for (edge_id_t e_i = 0; e_i < e_num; e_i++) {
            vertex_id_t &a = raw_edges[e_i].src;
            vertex_id_t &b = raw_edges[e_i].dst;
            while (a >= name2id.size()) {
                name2id.push_back(UINT_MAX);
            }
            while (b >= name2id.size()) {
                name2id.push_back(UINT_MAX);
            }
            if (name2id[a] == UINT_MAX) {
                name2id[a] = v_num++;
                degrees.push_back(0);
            }
            if (name2id[b] == UINT_MAX) {
                name2id[b] = v_num++;
                degrees.push_back(0);
            }
            a = name2id[a];
            b = name2id[b];
            degrees[a]++;
            if(as_undirected)
            {
                raw_edges[e_i + e_num].src = b;
                raw_edges[e_i + e_num].dst = a;
                degrees[b]++;
            }
        }

        e_num = raw_edges.size();

        LOG(WARNING) << block_mid_str(1) << "Read graph from files in " << timer.duration() << " seconds";
        LOG(WARNING) << block_mid_str(1) << "Vertices number: " << v_num;
        LOG(WARNING) << block_mid_str(1) << "Edges number: " << e_num;
        LOG(WARNING) << block_mid_str(1) << "As undirected: " << (as_undirected ? "true" : "false");

        //printf("degree %u\n", degrees[865222]);

        //Build CSR;
        AdjList* raw_adjlists;
        AdjUnit* raw_adjunits;
        //build_CSR(raw_adjlists, raw_adjunits, v_num, e_num, degrees);
        raw_adjlists = new AdjList[v_num];
        raw_adjunits = new AdjUnit[e_num];

        edge_id_t adjunits_pos = 0;

        //Build CSR, step 1: Fill adjlists;
        for(vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {
            raw_adjlists[v_i].degree = degrees[v_i];
            raw_adjlists[v_i].begin = raw_adjunits + adjunits_pos;
            adjunits_pos += degrees[v_i];
        }

        #pragma omp parallel for
        for(vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {
            degrees[v_i] = 0;
        }

        AdjUnit* tmp_unit;

        //Build CSR, step 2: Write adjunits;
        for(edge_id_t e_i = 0; e_i < e_num; e_i++)
        {
            vertex_id_t a = raw_edges[e_i].src;
            vertex_id_t b = raw_edges[e_i].dst;

            assert(degrees[a] < raw_adjlists[a].degree);

            tmp_unit = raw_adjlists[a].begin + degrees[a];
            tmp_unit->neighbor = b;
            degrees[a]++;
        }

        /*for(vertex_id_t e_i = 0; e_i < raw_adjlists[865222].degree; e_i++)
        {
            AdjUnit* tmpunit = raw_adjlists[865222].begin + e_i;
            printf("%lu ", tmpunit->neighbor);
        }
        printf("\n");*/

        //Get the number of edges of each subgraph
        std::ifstream ifs("/sys/devices/system/cpu/cpu0/cache/index3/size");
        std::string line;
        std::getline(ifs, line);
        unsigned int L3CacheSize = std::stoi(line);
        L3CacheSize = mtcfg.l2_cache_size;
        edge_per_subgraph = L3CacheSize * 90 / 4 / 100;
        uint64_t subgraph_size = mtcfg.l2_cache_size * 9 / 10;

        //Graph partition
        Timer partition_time;
        std::vector<std::vector<std::vector<vertex_id_t> > > disconnected_vertex_group;
        std::vector<std::vector<edge_id_t> > disconnected_vertex_group_edges;
        for(int i = 0; i < 2; i++)
        {
            std::vector<std::vector<vertex_id_t> > tmp_disconnected_group;
            std::vector<edge_id_t> tmp_group_edges;
            //std::vector<vertex_id_t> tmp_hub_vertex_group;
            disconnected_vertex_group.push_back(tmp_disconnected_group);
            disconnected_vertex_group_edges.push_back(tmp_group_edges);
            //hub_vertex_group.push_back(tmp_hub_vertex_group);
        }


        //printf("L3 cache size %u edges %lu\n", L3CacheSize, edge_per_subgraph);
        partition_id_t* partition_id_of_maxcut = new partition_id_t[v_num]();
        vertex_id_t* partition_vertex_num = new vertex_id_t[2]();
        edge_id_t* hyper_edge_num = new edge_id_t[2]();
        double* avg_deg = new double[2]();
        //hub_vertex_group.resize(2);
        default_rand_t* graphrd = mpool.alloc_new<default_rand_t>(1, 0);
        for(vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {
            if(raw_adjlists[v_i].degree >= bodary)
            {
                auto rdnum = graphrd->gen(2);
                hub_vertex_group.push_back(v_i);
                partition_id_of_maxcut[v_i] = rdnum + 3;
                partition_vertex_num[rdnum]++;
            }
            else
            {
                partition_id_of_maxcut[v_i] = 2;
            }
        }

        //printf("Max_cut begin\n");

        max_cut_partition(partition_id_of_maxcut, partition_vertex_num, raw_adjlists, raw_adjunits);
        //printf("hyper0: %u\n", partition_vertex_num[0]);
        //printf("hyper1: %u\n", partition_vertex_num[1]);

        std::vector<std::vector<VertexSortUnit> > Degree;
        Degree.push_back(std::vector<VertexSortUnit>() );
        Degree.push_back(std::vector<VertexSortUnit>() );

        edge_id_t inner_island_edges = 0;
        edge_id_t tot_low_edge = 0;
        edge_id_t tot_low_cut = 0;
        for(vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {
            vertex_id_t tmp_vertex = v_i;
            partition_id_t tmp_vertex_partition_id = partition_id_of_maxcut[tmp_vertex];
            VertexSortUnit vertex_degree;
            if(tmp_vertex_partition_id < 2)
            {
                vertex_degree.vertex = v_i;
                vertex_degree.degree = raw_adjlists[tmp_vertex].degree;
                Degree[tmp_vertex_partition_id].push_back(vertex_degree);
                hyper_edge_num[tmp_vertex_partition_id] += raw_adjlists[tmp_vertex].degree;
                /*
                for(vertex_id_t e_i = 0; e_i < raw_adjlists[tmp_vertex].degree; e_i++)
                {
                    AdjUnit* begin = raw_adjlists[tmp_vertex].begin + e_i;
                    auto vertex_neighbor = begin->neighbor;
                    auto vertex_partition_id = partition_id_of_maxcut[vertex_neighbor];
                    if(vertex_partition_id != tmp_vertex_partition_id)
                    {
                        tot_low_cut++; 
                    }
                    if(tmp_vertex_partition_id < 2)
                    {
                        tot_low_edge++;
                        if(vertex_partition_id != tmp_vertex_partition_id)
                        {
                            inner_island_edges++;
                        }
                    }
                }*/
            }
        }

        avg_deg[0] = double(hyper_edge_num[0] + hyper_edge_num[1]) / double(partition_vertex_num[0] + partition_vertex_num[1]);
        avg_deg[1] = double(hyper_edge_num[1]) / double(partition_vertex_num[1]);
        partition_id_t nump = ((hyper_edge_num[0] + hyper_edge_num[1]) * 4 + 8 * (partition_vertex_num[0] + partition_vertex_num[1])) / subgraph_size;
        if(nump == 0)
        {
            nump = 1;
        }
        vertex_id_t avg_vertex = (partition_vertex_num[0] + partition_vertex_num[1]) / nump;

        //printf("Max edge cuts + hub cuts %lu, tot low edge %lu, inner low cut %lu,  avg vertex %u, subgraph size %lu\n", tot_low_cut, tot_low_edge, inner_island_edges, avg_vertex, subgraph_size);

        for(int i = 0; i < 2; i++)
        {
            //printf("Start fat partition\n");
            dp_partition(Degree[i], disconnected_vertex_group[i], disconnected_vertex_group_edges[i], subgraph_size, avg_deg[0], avg_vertex);
            //Fat_partition(partition_id_of_maxcut, i, disconnected_vertex_group[i], disconnected_vertex_group_edges[i], raw_adjlists, raw_adjunits, partition_num);
        }

        printf("Partition time %lf s\n", partition_time.duration());

        vertex_id_t hub_vertices_num = 0;
        vertex_id_t low_degree_vertices_num = 0;
        edge_id_t hub_edges_num = 0;
        edge_id_t low_degree_edges = 0;

        for(partition_id_t p_i = 0; p_i < 2; p_i++)
        {
            for(vertex_id_t p_j = 0; p_j < disconnected_vertex_group[p_i].size(); p_j++)
            {
                low_degree_vertices_num += disconnected_vertex_group[p_i][p_j].size();
                low_degree_edges += disconnected_vertex_group_edges[p_i][p_j];
            }
        }

        hub_vertices_num = hub_vertex_group.size();
        for(vertex_id_t v_i = 0; v_i < hub_vertices_num; v_i++)
        {
            hub_edges_num += raw_adjlists[hub_vertex_group[v_i]].degree;
        }

        //printf("v_num e_num: %u %lu\n", v_num, e_num);
        //printf("hub_v_num hub_e_num: %u %lu\n", hub_vertices_num, hub_edges_num);
        //printf("low_v_num low_e_num: %u %lu\n", low_degree_vertices_num, low_degree_edges);



        //exit(-1);
    
        std::random_shuffle(hub_vertex_group.begin(), hub_vertex_group.end());

        std::vector<std::vector<vertex_id_t>> combined_vertex_id;

        for(uint64_t g_i = 0; g_i < disconnected_vertex_group.size(); g_i++)
        {
            for(partition_id_t p_i = 0; p_i < disconnected_vertex_group[g_i].size(); p_i++)
            {
                combined_vertex_id.push_back(disconnected_vertex_group[g_i][p_i]);
            }
        }



        //printf("reorder begin\n");
        vertex_id_t max_vertices_of_dir_subgraph = 0;

        for(partition_id_t p_i = 0; p_i < combined_vertex_id.size(); p_i++)
        {
            if(combined_vertex_id[p_i].size() > max_vertices_of_dir_subgraph)
            {
                max_vertices_of_dir_subgraph = combined_vertex_id[p_i].size();
            }
        }

        fat_subgraph_pos = hub_vertices_num;

        vertices_per_pre_subgraph = mtcfg.l2_cache_size * 9 / 10 / (sizeof(edge_id_t) + 2 * sizeof(vertex_id_t) + 64) - 1;
        if(fat_subgraph_pos != 0)
        {
            partition_num_pre_subgraph = fat_subgraph_pos / vertices_per_pre_subgraph + 1;
            if(partition_num_pre_subgraph % 2 == 1)
            {
                partition_num_pre_subgraph++;
            }
            partition_num_pre_subgraph_left = partition_num_pre_subgraph / 2;
            if(partition_num_pre_subgraph_left % mtcfg.thread_num != 0)
            {
                partition_num_pre_subgraph_left = (partition_num_pre_subgraph_left / mtcfg.thread_num + 1) * mtcfg.thread_num;
            }
            partition_num_pre_subgraph = partition_num_pre_subgraph_left * 2;
        }
        else
        {
            partition_num_pre_subgraph = 0;
            partition_num_pre_subgraph_left = 0;
        }
        if(partition_num_pre_subgraph != 0)
        {
            vertices_per_pre_subgraph = fat_subgraph_pos / partition_num_pre_subgraph + 1;
        }
        else
        {
            vertices_per_pre_subgraph = 0;
        }

        partition_num_low_left = partition_num_pre_subgraph + disconnected_vertex_group[0].size();
        vertex_id_t reorderedidmax = fat_subgraph_pos + max_vertices_of_dir_subgraph * combined_vertex_id.size();
        reorderid = new vertex_id_t[v_num]();
        vertex_id_t reorderedid = 0;
        partition_num = 0;
        vertex_partition_id = new partition_id_t[reorderedidmax]();

        //#pragma omp parallel for
        for(uint64_t v_i = 0; v_i < hub_vertex_group.size(); v_i++)
        {
            reorderid[hub_vertex_group[v_i]] = reorderedid;
            vertex_partition_id[reorderedid] = partition_num + v_i / vertices_per_pre_subgraph;
            reorderedid++; 
        }

        assert(fat_subgraph_pos <= vertices_per_pre_subgraph * partition_num_pre_subgraph);
        partition_num += partition_num_pre_subgraph;
        //partition_sampler_class.push_back(ClassExclusiveBufferSampler);
        for(partition_id_t p_i = 0; p_i < partition_num_pre_subgraph; p_i++)
        {
            partition_begin.push_back(p_i * vertices_per_pre_subgraph);
            partition_end.push_back(std::min((p_i + 1) * vertices_per_pre_subgraph, fat_subgraph_pos));
            partition_sampler_class.push_back(ClassExclusiveBufferSampler);
        }
        vertex_id_t start_pos = fat_subgraph_pos;

        vertices_per_dir_subgraph = max_vertices_of_dir_subgraph;
        //printf("vertices per subgraph %u %u\n", vertices_per_pre_subgraph, vertices_per_dir_subgraph);
        //vertices_per_pre_subgraph = vertices_per_pre_subgraph;

        //printf("reorder begin2\n");
        

        for(vertex_id_t s_i = 0; s_i < combined_vertex_id.size(); s_i++)
        {
            //printf("subgraph %u , number of vertices %lu\n", s_i, combined_vertex_id[s_i].size());
            for(vertex_id_t v_i = 0; v_i < combined_vertex_id[s_i].size(); v_i++)
            {
                assert(combined_vertex_id[s_i][v_i] < v_num);
                reorderid[combined_vertex_id[s_i][v_i]] = reorderedid;
                assert(reorderedid < reorderedidmax);
                vertex_partition_id[reorderedid] = partition_num;
                reorderedid++;
            }
            for(vertex_id_t v_i = combined_vertex_id[s_i].size(); v_i < max_vertices_of_dir_subgraph; v_i++)
            {
                reorderedid++;
                vertex_partition_id[reorderedid] = partition_num;
            }
            //printf("\n");
            partition_num++;
            partition_sampler_class.push_back(ClassDirectSampler);
            partition_begin.push_back(start_pos);
            partition_end.push_back(start_pos + combined_vertex_id[s_i].size());
            //start_pos += combined_vertex_id[s_i].size();
            start_pos += max_vertices_of_dir_subgraph;
            //partition_end.push_back(start_pos);
        }

        if(reorderedid != fat_subgraph_pos + max_vertices_of_dir_subgraph * combined_vertex_id.size())
        {
            printf("some vertices is wrong %u\n", reorderedid);
        }

        //printf("reordered %u %u %lu\n", reorderedid, v_num, e_num);

        //delete []raw_adjlists;
        delete []raw_adjunits;

        degrees.resize(reorderedid);
        #pragma omp parallel for
        for(vertex_id_t v_i = 0; v_i < reorderedid; v_i++)
        {
            degrees[v_i] = 0;
        }

        //printf("after\n\n");

        //raw_edges
        #pragma omp parallel for
        for(edge_id_t e_i = 0; e_i < raw_edges.size(); e_i++)
        {
            raw_edges[e_i].src = reorderid[raw_edges[e_i].src];
            raw_edges[e_i].dst = reorderid[raw_edges[e_i].dst];
            degrees[raw_edges[e_i].src]++;
        }
        //vertex_id_t* degreesReo = new vertex_id_t[reorderedid];

        //printf("init degree\n");

        #pragma omp parallel for
        for(vertex_id_t v_i = 0; v_i < v_num; v_i++)
        {
            //id2name[]
            // TBD name2id & id2name;
            assert(degrees[reorderid[v_i]] == raw_adjlists[v_i].degree);
        }

        //printf("build csr start\n");

        delete []raw_adjlists;
        delete []reorderid;
        std::vector<vertex_id_t>().swap(name2id);

        //printf("build csr start\n");
        v_num_old = v_num;
        v_num = reorderedid;

        //printf("reordered %u %u\n", reorderedid, v_num);

        //build_CSR(adjlists, adjunits, v_num, e_num, degrees);
        //adjlists = new AdjList1[v_num];
        //printf("v_num %u\n", v_num);
        //adjunits = new AdjUnit[e_num];
        //printf("e_num %lu\n", e_num);

        socket_partition_nums.reset(new int[mtcfg.socket_num]);
        for (int s_i = 0; s_i < mtcfg.socket_num; s_i++) {
            socket_partition_nums[s_i] = 0;
        }
        partition_socket.resize(this->partition_num);
        for (partition_id_t p_i = 0; p_i < this->partition_num; p_i++) {
            if (p_i % (mtcfg.socket_num * 2) < partition_id_t(mtcfg.socket_num)) {
                partition_socket[p_i] = p_i % mtcfg.socket_num;
            } else {
                partition_socket[p_i] = mtcfg.socket_num - p_i % mtcfg.socket_num - 1;
            }
            socket_partition_nums[partition_socket[p_i]]++;
        }
        socket_partitions.reset(new int*[mtcfg.socket_num]);
        std::vector<vertex_id_t> temp_socket_partition_count(mtcfg.socket_num, 0);
        for (int s_i = 0; s_i < mtcfg.socket_num; s_i++) {
            socket_partitions[s_i] = mpool.alloc<int>(socket_partition_nums[s_i], s_i);
        }
        for (partition_id_t p_i = 0; p_i < this->partition_num; p_i++) {
            auto socket = partition_socket[p_i];
            socket_partitions[socket][temp_socket_partition_count[socket]++] = p_i;
        }

        //printf("init socket\n");


        //printf("build csr start\n");

        adjlists.reset(new AdjList*[mtcfg.socket_num]);
        for (int s_i = 0; s_i < mtcfg.socket_num; s_i++) {
            adjlists[s_i] = mpool.alloc<AdjList>(v_num, s_i);
        }
        #pragma omp parallel for
        for (vertex_id_t v_i = 0; v_i < v_num; v_i++) {
            for (int s_i = 0; s_i < mtcfg.socket_num; s_i++) {
                adjlists[s_i][v_i].degree = degrees[v_i];
            }
        }

        std::vector<vertex_id_t>().swap(degrees);

        edges.reset(new AdjUnit*[mtcfg.socket_num]);
        for (int s_i = 0; s_i < mtcfg.socket_num; s_i++) {
            edge_id_t p_e_num = 0;
            #pragma omp parallel for reduction (+: p_e_num)
            for (int p_i = 0; p_i < socket_partition_nums[s_i]; p_i++) {
                auto partition = socket_partitions[s_i][p_i];
                for (vertex_id_t v_i = partition_begin[partition]; v_i < partition_end[partition]; v_i++) {
                    p_e_num += adjlists[s_i][v_i].degree;
                }
            }
            edges[s_i] = mpool.alloc<AdjUnit>(p_e_num, s_i);
        }
        std::vector<AdjUnit*> edge_end(v_num);
        for (int s_i = 0; s_i < mtcfg.socket_num; s_i++) {
            size_t temp = 0;
            for (int p_i = 0; p_i < socket_partition_nums[s_i]; p_i++) {
                auto partition = socket_partitions[s_i][p_i];
                for (vertex_id_t v_i = partition_begin[partition]; v_i < partition_end[partition]; v_i++) {
                    adjlists[0][v_i].begin = edges[s_i] + temp;
                    edge_end[v_i] = edges[s_i] + temp;
                    temp += adjlists[s_i][v_i].degree;
                }
            }
        }

        #pragma omp parallel for
        for (size_t e_i = 0; e_i < raw_edges.size(); e_i++) {
            vertex_id_t u = raw_edges[e_i].src;
            vertex_id_t v = raw_edges[e_i].dst;
            auto *temp = __sync_fetch_and_add(&edge_end[u], sizeof(AdjUnit));
            temp->neighbor = v;
            if (as_undirected) {
                auto *temp = __sync_fetch_and_add(&edge_end[v], sizeof(AdjUnit));
                temp->neighbor = u;
            }
        }
        for (int s_i = 1; s_i < mtcfg.socket_num; s_i++) {
            #pragma omp parallel for
            for (vertex_id_t v_i = 0; v_i < v_num; v_i++) {
                adjlists[s_i][v_i] = adjlists[0][v_i];
            }
        }
        partition_edge_num.resize(partition_num);

        #pragma omp parallel for
        for(partition_id_t p_i = 0; p_i < partition_num; p_i++)
        {
            edge_id_t p_edge = 0;
            for(vertex_id_t v_i = partition_begin[p_i]; v_i < partition_end[p_i]; v_i++)
            {
                p_edge += adjlists[0][v_i].degree;
            }
            partition_edge_num[p_i] = p_edge;
        }
        //printf("end load graph\n");

        std::vector<Edge>().swap(raw_edges);
    }

    // Create bloom filter for node2vec
    void prepare_neighbor_query() {
        Timer timer;
        #pragma omp parallel for schedule(dynamic, 1)
        for (partition_id_t p_i = 0; p_i < partition_num; p_i++) {
            for (vertex_id_t v_i = partition_begin[p_i]; v_i < partition_end[p_i]; v_i++) {
                AdjList* adj= adjlists[0] + v_i;
                std::sort(adj->begin, adj->begin + adj->degree, [](const AdjUnit& a, const AdjUnit& b){return a.neighbor < b.neighbor;});
            }
        }
        bf.reset(new BloomFilter(mtcfg));
        bf->create(as_undirected ? e_num / 2 : e_num);
        #pragma omp parallel for schedule(dynamic, 1)
        for (partition_id_t p_i = 0; p_i < partition_num; p_i++) {
            for (vertex_id_t v_i = partition_begin[p_i]; v_i < partition_end[p_i]; v_i++) {
                AdjList* adj = adjlists[0] + v_i;
                for (auto *edge = adj->begin; edge < adj->begin + adj->degree; edge++) {
                    bf->insert(v_i, edge->neighbor);
                }
            }
        }
        LOG(WARNING) << block_mid_str() << "Prepare neighborhood query in " << timer.duration() << " seconds";
    }

    // Neighborhood query for node2vec
    bool has_neighbor(vertex_id_t src, vertex_id_t dst, int socket) {
        if (bf->exist(src, dst) == false) {
            return false;
        }
        AdjList* adj = adjlists[socket] + src;
        AdjUnit unit;
        unit.neighbor = dst;
        return std::binary_search(adj->begin, adj->begin + adj->degree, unit, [](const AdjUnit &a, const AdjUnit &b) { return a.neighbor < b.neighbor; });
    }

    size_t get_memory_size() {
        return sizeof(AdjList) * v_num * (size_t) mtcfg.socket_num + sizeof(AdjUnit) * e_num;
    }

    size_t get_csr_size() {
        return sizeof(AdjList) * v_num + sizeof(AdjUnit) * e_num;
    }
};
