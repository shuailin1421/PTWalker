template<typename sampler_t>
    walker_id_t walk_message(sampler_t *sampler, int w_i, partition_id_t p_i) {
        int worker_id = omp_get_thread_num();
        auto *rd = this->rands[worker_id];
        auto used_pool_start = used_pool_pos[w_i][p_i].walks_start;
        auto used_pool_end = used_pool_pos[w_i][p_i].walks_end;
        used_pool_pos[w_i][p_i].walks_start = used_pool_pos[w_i][p_i].walks_end - 1;
        partition_id_t tmp_partition;
        vertex_id_t vertex_old;
        vertex_id_t vertex_new;
        walker_id_t finished_walker = 0;
        walker_id_t last_computing_walker = 0;
        uint16_t last_walker_start_recore;
        uint16_t last_walker_end_recore;
        uint64_t *vertex_recore = new uint64_t[used_pool_end * walks_per_container];
        walker_id_t *poolid_recore = new walker_id_t[used_pool_end];
        walker_id_t poolid;
        //if(worker_id == 0)
        //printf("%d %d %u start compute walkers\n", worker_id, w_i, p_i);


        for(walker_id_t wp_i = used_pool_start; wp_i < used_pool_end; wp_i++)
        {
            used_pool_lock[w_i][p_i].lock();
            poolid = used_pool[w_i][p_i][wp_i];
            used_pool_lock[w_i][p_i].unlock();
            poolid_recore[wp_i] = poolid;
            uint16_t walker_start = walker_pos[w_i][poolid].walks_start;
            uint16_t walker_end = walker_pos[w_i][poolid].walks_end;
            walker_pos[w_i][poolid].walks_start = walker_pos[w_i][poolid].walks_end;
            last_walker_start_recore = walker_start;
            last_walker_end_recore = walker_end;
            
            for(uint16_t wk_i = walker_start; wk_i < walker_end; wk_i++)
            {
                vertex_old = walkers[w_i][poolid][wk_i].w_data & 0x000000000fffffff;
                if(vertex_old >= graph->v_num)
                {
                    //printf("error vertex old %u %u %u\n", vertex_old, poolid, p_i);
                    finished_walker++;
                    walkers[w_i][poolid][wk_i].w_data = 0xffffffffffffffff;
                    continue;
                    //exit(-1);
                }
                tmp_partition = graph->vertex_partition_id_fuc(vertex_old);
                if(tmp_partition != p_i)
                {
                    //printf("error! %u tmp_partition %u != p_i %u\n", poolid, tmp_partition, p_i);
                    finished_walker++;
                    walkers[w_i][poolid][wk_i].w_data = 0xffffffffffffffff;
                    continue;
                    //exit(-1);
                }
                last_computing_walker = wp_i * walks_per_container + wk_i;
                //if(worker_id == 0)
                //printf("%d %u vertex %u\n",worker_id, wp_i * 256 + wk_i, vertex_old);
                //assert(vertex_old < graph->v_num);
                //tmp_partition = p_i;
                while(true)
                {
                    vertex_new = sampler->sample(vertex_old, rd);
                    //vertex_recore[last_computing_walker] = vertex_new;
                    //if(worker_id == 0)
                    //printf("%d ->%u vertex new %u\n",worker_id, wp_i * 256 + wk_i, vertex_new);
                    walkers[w_i][poolid][wk_i].w_data = walkers[w_i][poolid][wk_i].w_data - vertex_old + vertex_new + 0x0100000000000000;
                    vertex_recore[last_computing_walker] = walkers[w_i][poolid][wk_i].w_data;
                    vertex_old = vertex_new;
                    if(rd->gen(10) > 0)
                    {
                        tmp_partition = graph->vertex_partition_id_fuc(vertex_new);
                        if(tmp_partition != p_i)
                        {
                            break;
                        }
                        //printf(" ->%d partition id %u\n", worker_id, tmp_partition);
                    }
                    else
                    {
                        finished_walker++;
                        walkers[w_i][poolid][wk_i].w_data = 0xffffffffffffffff;
                        break;
                    }
                }
            }
        }
        walker_id_t unused_pool_size;
        if(unused_pool_pos[worker_id].pool_end < unused_pool_pos[worker_id].pool_start)
        {
            unused_pool_size = unused_pool_pos[worker_id].pool_end + walks_units_max_size - unused_pool_pos[worker_id].pool_start;
        }
        else
        {
            unused_pool_size = unused_pool_pos[worker_id].pool_end - unused_pool_pos[worker_id].pool_start;
        }
        walker_id_t used_pool_size = 0;
        for(partition_id_t pp_i = 0; pp_i < graph->partition_num; pp_i++)
        {
            used_pool_size += used_pool_pos[worker_id][pp_i].walks_end - used_pool_pos[worker_id][pp_i].walks_start;
        }
        if(unused_pool_size + used_pool_size != walks_units_size[worker_id])
        {
            printf("error! %u %u %u\n", unused_pool_size, used_pool_size, walks_units_size[worker_id]);
            //exit(-1);
        }
        //if(worker_id == 0)
        //printf("%d %d %u start move walkers\n",worker_id ,w_i, p_i);
        walker_id_t finished_walker_test = 0;

        uint16_t recore_pi = 0;
        walker_id_t mpoolid;
        walker_id_t new_poolid;
        uint16_t wk_i = 0;
        //printf(" -> walk move start %u end %u\n", used_pool_start, used_pool_end);
        for(uint16_t up_i = used_pool_start; up_i < used_pool_end; up_i++)
        {
            
            recore_pi = up_i;
            used_pool_lock[w_i][p_i].lock();
            poolid = used_pool[w_i][p_i][up_i];
            used_pool_lock[w_i][p_i].unlock();
            if(poolid != poolid_recore[up_i])
            {
                printf("error poolid\n");
                exit(-1);
            }
            uint16_t walker_start = walker_pos[w_i][poolid].walks_start;
            uint16_t walker_end = walker_pos[w_i][poolid].walks_end;
            if(up_i == used_pool_end - 1)
            {
                walker_start = last_walker_start_recore;
                walker_end = last_walker_end_recore;
            }
            bool if_locl_walker = false;
            for(wk_i = walker_start; wk_i < walker_end; wk_i++)
            {
                //if(worker_id == 0)
                //printf("%d %u vertex %u\n",worker_id, up_i * 256 + wk_i, walkers[w_i][poolid][wk_i].w_data & 0x000000000fffffff);
                if(walkers[w_i][poolid][wk_i].w_data != 0xffffffffffffffff)
                {
                    vertex_old = walkers[w_i][poolid][wk_i].w_data & 0x000000000fffffff;
                    tmp_partition = graph->vertex_partition_id_fuc(vertex_old);
                    //move walks
                    if(tmp_partition == p_i)
                    {
                        //printf("%d -> cur partition %u %u\n",worker_id, last_computing_walker, up_i * walks_per_container + wk_i);
                        printf("nor right\n");
                        if(walkers[w_i][poolid][wk_i].w_data != vertex_recore[up_i * walks_per_container + wk_i])
                        {
                            printf("%d %d (%u %u)%u %u %u The move walker is wrong! %u %u %u %u\n",worker_id, w_i, tmp_partition, p_i, used_pool_start, used_pool_end, up_i, vertex_old, vertex_recore[up_i * walks_per_container + wk_i] & 0x000000000fffffff, up_i * walks_per_container + wk_i, last_computing_walker);
                        }
                        continue;
                        //up_i = used_pool_end;
                        //walker_pos[w_i][poolid].walks_start = wk_i;
                        //if_locl_walker = true;
                        //break;
                    }
                    else
                    {
                        //move this walker
                        if(walkers[w_i][poolid][wk_i].w_data != vertex_recore[up_i * walks_per_container + wk_i])
                        {
                            printf("%d %d (%u %u)%u %u %u The move walker is wrong! %u %u %u %u\n",worker_id, w_i, tmp_partition, p_i, used_pool_start, used_pool_end, up_i, vertex_old, vertex_recore[up_i * walks_per_container + wk_i] & 0x000000000fffffff, up_i * walks_per_container + wk_i, last_computing_walker);
                        }
                        mpoolid = used_pool_pos[worker_id][tmp_partition].rpoolid;
                        /*if(mpoolid >= walks_units_size[worker_id])
                        {
                            printf("mpoolid error\n");
                            exit(-1);
                        }*/
                        uint16_t &walkers_pos_end = walker_pos[worker_id][mpoolid].walks_end;
                        /*if(walkers_pos_end > walks_per_container)
                        {
                            printf("walk pos end error\n");
                            exit(-1);
                        }*/
                        if(walker_pos[worker_id][mpoolid].walks_end < walks_per_container)
                        {
                            walkers[worker_id][mpoolid][walkers_pos_end].w_data = walkers[w_i][poolid][wk_i].w_data;
                            walkers_pos_end++;
                            //if(worker_id == 0)
                            //printf("%d -> move to %d %u pool id %u %u\n",worker_id, worker_id, tmp_partition, mpoolid,  walkers_pos_end);
                        }
                        else
                        {
                            unused_pool_add_lock[worker_id].lock();
                            if(unused_pool_pos[worker_id].pool_start == unused_pool_pos[worker_id].pool_end)
                            {
                                printf("No pool left generate new walker array\n");
                                walker_id_t pool_size_left = walks_units_max_size - walks_units_size[worker_id];
                                walker_id_t generated_pool_size = walks_units_max_size / 16;
                                walker_id_t generated_pool_end = walks_units_size[worker_id] + generated_pool_size;
                                unused_pool_pos[worker_id].pool_start = 0;
                                unused_pool_pos[worker_id].pool_end = 0;
                                for(walker_id_t gpid = walks_units_size[worker_id]; gpid < generated_pool_end; gpid++)
                                {
                                    walkers[worker_id][gpid] = mpool.alloc_new<walker_t>(walks_per_container, 0);
                                    unused_pool[worker_id][unused_pool_pos[worker_id].pool_end] = gpid;
                                    unused_pool_pos[worker_id].pool_end++;
                                }
                                walks_units_size[worker_id] += generated_pool_size;
                                printf("new generate %u arrays\n",generated_pool_size);
                            }
                            unused_pool_add_lock[worker_id].unlock();
                            new_poolid = unused_pool[worker_id][unused_pool_pos[worker_id].pool_start];
                            /*if(new_poolid >= walks_units_size[worker_id])
                            {
                                printf("mpoolid error\n");
                                exit(-1);
                            }*/
                            unused_pool_pos[worker_id].pool_start = (unused_pool_pos[worker_id].pool_start + 1) % walks_units_max_size;
                            while(walker_pos[worker_id][new_poolid].walks_start != 0 || walker_pos[worker_id][new_poolid].walks_end != 0)
                            {
                                printf("error not zero %u %u\n", walker_pos[worker_id][new_poolid].walks_start, walker_pos[worker_id][new_poolid].walks_end);
                                new_poolid = unused_pool[worker_id][unused_pool_pos[worker_id].pool_start];
                                unused_pool_pos[worker_id].pool_start = (unused_pool_pos[worker_id].pool_start + 1) % walks_units_max_size;
                                //exit(-1);
                            }
                            walker_pos[worker_id][new_poolid].walks_start = 0;
                            walker_pos[worker_id][new_poolid].walks_end = 0;
                            used_pool_lock[worker_id][tmp_partition].lock();
                            if(used_pool_pos[worker_id][tmp_partition].walks_end >= used_pool_pos[worker_id][tmp_partition].walks_size)
                            {
                                //if(worker_id == 0)
                                printf("walker size double\n");
                                walker_id_t* new_pool = new walker_id_t[used_pool_pos[worker_id][tmp_partition].walks_size * 2];
                                for(walker_id_t wp_i = 0; wp_i < used_pool_pos[worker_id][tmp_partition].walks_size; wp_i++)
                                {
                                    new_pool[wp_i] = used_pool[worker_id][tmp_partition][wp_i];
                                }
                                delete[] used_pool[worker_id][tmp_partition];
                                used_pool[worker_id][tmp_partition] = new_pool;
                                used_pool_pos[worker_id][tmp_partition].walks_size = used_pool_pos[worker_id][tmp_partition].walks_size * 2;
                            }
                            used_pool[worker_id][tmp_partition][used_pool_pos[worker_id][tmp_partition].walks_end] = new_poolid;
                            used_pool_pos[worker_id][tmp_partition].walks_end++;
                            used_pool_pos[worker_id][tmp_partition].rpoolid = new_poolid;
                            used_pool_lock[worker_id][tmp_partition].unlock();
                            walkers[worker_id][new_poolid][walker_pos[worker_id][new_poolid].walks_end].w_data = walkers[w_i][poolid][wk_i].w_data;
                            walker_pos[worker_id][new_poolid].walks_end++;
                            //if(worker_id == 0)
                            //printf("%d -> new move to %d %u pool id %u %u\n",worker_id, worker_id, tmp_partition, new_poolid,  walker_pos[worker_id][new_poolid].walks_end);
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
            printf("recore pi %u poolid %u\n", used_pool[w_i][p_i][0], poolid);
            used_pool_pos[w_i][p_i].walks_start = 0;
            used_pool_pos[w_i][p_i].walks_end = upcount;
            used_pool_lock[w_i][p_i].unlock();
        }
        else
        {
            printf("%u recore pi %u used_pool_start %u used_pool_end %u\n", p_i, recore_pi, used_pool_start, used_pool_end);
        }

        return finished_walker;
    }