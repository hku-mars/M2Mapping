#include "rog_map_cuda/config.cuh"


namespace rog_map {
        __host__ void ROSParamLoader::resetMapSize(ROGMapConfig &cfg){
            int inflation_ratio = ceil( cfg.inflation_resolution/cfg.resolution );
            cfg.inflation_resolution = cfg.resolution * inflation_ratio;
            cfg.half_map_size_d = cfg.map_size_d / 2;
            cfg.inf_half_map_size_i = (1.0/cfg.inflation_resolution *  cfg.half_map_size_d  +  cfg.half_map_size_d .cwiseSign() * 0.5).cast<int>();
            cfg.half_map_size_d = cfg.inf_half_map_size_i.cast<double>() * cfg.inflation_resolution;
            cfg.map_size_d = cfg.half_map_size_d * 2;
            cfg.half_map_size_i = cfg.inf_half_map_size_i * inflation_ratio;
            cfg.inf_half_map_size_i += Vec3i::Constant(cfg.inflation_step);

            cfg.half_local_update_box_d = cfg.local_update_box_d / 2;
            cfg.half_local_update_box_i = (1.0/cfg.resolution *  cfg.half_local_update_box_d  +  Vec3f::Constant(0.5)).cast<int>();
            cfg.local_update_box_i = cfg.half_local_update_box_i * 2 + Vec3i::Constant(1);
        }

        __host__ void ROSParamLoader::initConfig(ROGMapConfig &cfg)  {
            resetMapSize(cfg);

            /// Probabilistic Update
            #define logit(x) (log((x) / (1 - (x))))
            cfg.l_hit = logit(cfg.p_hit);
            cfg.l_miss = logit(cfg.p_miss);
            cfg.l_min = logit(cfg.p_min);
            cfg.l_max = logit(cfg.p_max);
            cfg.l_occ = logit(cfg.p_occ);
            cfg.l_free = logit(cfg.p_free);

            printf(" -- [ROG] Configurations:\n");
            printf("\tl_min: %0.3f\n", cfg.l_min);
            printf("\tl_miss: %0.3f\n", cfg.l_miss);
            printf( "\tl_free: %0.3f\n", cfg.l_free);
            printf("\tl_occ: %0.3f\n", cfg.l_occ);
            printf("\tl_hit: %0.3f\n", cfg.l_hit);
            printf("\tl_max: %0.3f\n", cfg.l_max);


            cfg.half_map_size_d = cfg.map_size_d / 2.0;

            // init spherical neighbor
            for (int dx = -cfg.inflation_step; dx <= cfg.inflation_step; dx++) {
                for (int dy = -cfg.inflation_step; dy <= cfg.inflation_step; dy++) {
                    for (int dz = -cfg.inflation_step; dz <= cfg.inflation_step; dz++) {
                        if (dx * dx + dy * dy + dz * dz <= cfg.inflation_step * cfg.inflation_step) {
                            cfg.spherical_neighbor.push_back(Vec3i(dx, dy, dz));
                        }
                    }
                }
            }
            std::sort(cfg.spherical_neighbor.begin(), cfg.spherical_neighbor.end(), [](const Vec3i &a, const Vec3i &b) {
                return a.norm() < b.norm();
            });
            cfg.spherical_neighbor_N = cfg.spherical_neighbor.size();
            printf(" -- [ROG] Spherical neighbor size: %lu\n", cfg.spherical_neighbor.size());
        }
}

