MergedProfile 
     Fragments:
       Fragment 0:
         Pipeline : 0(instance_num=1):
            - WaitWorkerTime: avg 25.500us, max 25.500us, min 25.500us
           RESULT_SINK_OPERATOR (id=0):
              - CloseTime: avg 4.804us, max 4.804us, min 4.804us
              - ExecTime: avg 201.673us, max 201.673us, min 201.673us
              - InitTime: avg 11.993us, max 11.993us, min 11.993us
              - InputRows: sum 100, avg 100, max 100, min 100
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 24.920us, max 24.920us, min 24.920us
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
              - WaitForDependency[RESULT_SINK_OPERATOR_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
           EXCHANGE_OPERATOR (id=22):
              - PlanInfo
                 - offset: 0
                 - limit: 100
              - BlocksProduced: sum 1, avg 1, max 1, min 1
              - CloseTime: avg 16.109us, max 16.109us, min 16.109us
              - ExecTime: avg 203.908us, max 203.908us, min 203.908us
              - InitTime: avg 117.681us, max 117.681us, min 117.681us
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 80.00 KB, avg 80.00 KB, max 80.00 KB, min 80.00 KB
              - OpenTime: avg 19.116us, max 19.116us, min 19.116us
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 100, avg 100, max 100, min 100
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForData0: avg 2sec658ms, max 2sec658ms, min 2sec658ms
       Fragment 1:
         Pipeline : 0(instance_num=48):
            - WaitWorkerTime: avg 16.449us, max 82.481us, min 4.877us
           DATA_STREAM_SINK_OPERATOR (dest_id=22):
              - BlocksProduced: sum 48, avg 1, max 1, min 1
              - CloseTime: avg 491ns, max 1.451us, min 258ns
              - ExecTime: avg 40.808us, max 154.511us, min 13.729us
              - InitTime: avg 29.951us, max 123.2us, min 9.988us
              - InputRows: sum 100, avg 2, max 100, min 0
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 3.626us, max 15.787us, min 874ns
              - OverallThroughput: sum 0.0 /sec, avg 0.0 /sec, max 0.0 /sec, min 0.0 /sec
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForRpcBufferQueue: avg 0ns, max 0ns, min 0ns
           LOCAL_EXCHANGE_OPERATOR (LOCAL_MERGE_SORT) (id=-5):
              - BlocksProduced: sum 1, avg 0, max 1, min 0
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 51.744us, max 2.392ms, min 658ns
              - GetBlockFailedTime: sum 0, avg 0, max 0, min 0
              - InitTime: avg 5.211us, max 194.674us, min 491ns
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 288.00 KB, avg 6.00 KB, max 6.00 KB, min 6.00 KB
              - OpenTime: avg 0ns, max 0ns, min 0ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 100, avg 2, max 100, min 0
              - WaitForDependencyTime: avg 2sec655ms, max 2sec655ms, min 2sec655ms
                - WaitForData0: avg 2sec655ms, max 2sec655ms, min 2sec655ms
                - WaitForData2: avg 502.108us, max 502.108us, min 502.108us
                - WaitForData42: avg 174.158us, max 174.158us, min 174.158us
              - WaitForDependency[LOCAL_MERGE_EXCHANGE_OPERATOR_DEPENDENCY]Time: avg 318.567us, max 798.722us, min 0ns
         Pipeline : 1(instance_num=48):
            - WaitWorkerTime: avg 22.91us, max 51.987us, min 10.951us
           LOCAL_EXCHANGE_SINK_OPERATOR (LOCAL_MERGE_SORT) (id=-5):
              - CloseTime: avg 404ns, max 5.711us, min 121ns
              - ExecTime: avg 6.900us, max 14.148us, min 3.949us
              - InitTime: avg 2.166us, max 6.515us, min 955ns
              - InputRows: sum 4.8K (4800), avg 100, max 100, min 100
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 232ns, max 1.163us, min 39ns
              - WaitForDependency[LOCAL_EXCHANGE_SINK_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
           SORT_OPERATOR (id=21 , nereids_id=3112):
              - PlanInfo
                 - order by: c_customer_id ASC
                 - TOPN OPT
                 - algorithm: heap sort
                 - offset: 0
                 - limit: 100
              - BlocksProduced: sum 48, avg 1, max 1, min 1
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 872ns, max 1.835us, min 300ns
              - InitTime: avg 0ns, max 0ns, min 0ns
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 0ns, max 0ns, min 0ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 4.8K (4800), avg 100, max 100, min 100
              - WaitForDependency[SORT_OPERATOR_DEPENDENCY]Time: avg 2sec654ms, max 2sec656ms, min 2sec653ms
         Pipeline : 2(instance_num=48):
            - WaitWorkerTime: avg 2.328ms, max 5.67ms, min 53.6us
           SORT_SINK_OPERATOR (id=21 , nereids_id=3112):
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 6.379ms, max 8.777ms, min 3.774ms
              - InitTime: avg 4.681us, max 44.879us, min 1.274us
              - InputRows: sum 172.81K (172810), avg 3.6K (3600), max 4.217K (4217), min 3.135K (3135)
              - MemoryUsage: sum 850.88 KB, avg 17.73 KB, max 20.88 KB, min 15.75 KB
              - MemoryUsagePeak: sum 850.88 KB, avg 17.73 KB, max 20.88 KB, min 15.75 KB
              - MemoryUsageSortBlocks: sum 850.88 KB, avg 17.73 KB, max 20.88 KB, min 15.75 KB
              - OpenTime: avg 51.217us, max 269.26us, min 4.800us
              - WaitForDependency[SORT_SINK_OPERATOR_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
           LOCAL_EXCHANGE_OPERATOR (PASSTHROUGH) (id=-4):
              - BlocksProduced: sum 2.304K (2304), avg 48, max 48, min 48
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 125.966us, max 181.801us, min 76.31us
              - GetBlockFailedTime: sum 41, avg 0, max 5, min 0
              - InitTime: avg 1.265us, max 8.240us, min 497ns
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 2.90 MB, avg 61.86 KB, max 107.50 KB, min 19.50 KB
              - OpenTime: avg 0ns, max 0ns, min 0ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 172.81K (172810), avg 3.6K (3600), max 4.217K (4217), min 3.135K (3135)
              - WaitForDependency[LOCAL_EXCHANGE_OPERATOR_DEPENDENCY]Time: avg 2sec645ms, max 2sec647ms, min 2sec645ms
         Pipeline : 3(instance_num=48):
            - WaitWorkerTime: avg 1.84ms, max 4.668ms, min 25.981us
           LOCAL_EXCHANGE_SINK_OPERATOR (PASSTHROUGH) (id=-4):
              - CloseTime: avg 1.107us, max 19.883us, min 223ns
              - ExecTime: avg 451.11us, max 2.941ms, min 95.877us
              - InitTime: avg 4.582us, max 44.770us, min 886ns
              - InputRows: sum 172.81K (172810), avg 3.6K (3600), max 3.724K (3724), min 3.445K (3445)
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 144ns, max 704ns, min 40ns
              - WaitForDependency[LOCAL_EXCHANGE_SINK_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
           HASH_JOIN_OPERATOR (id=20 , nereids_id=3102):
              - PlanInfo
                 - join op: INNER JOIN(PARTITIONED)[]
                 - equal join conjunct: (c_customer_sk = ctr_customer_sk)
                 - runtime filters: RF006[min_max] <- ctr_customer_sk(516012/524288/1048576), RF007[in_or_bloom] <- ctr_customer_sk(516012/524288/1048576)
                 - cardinality=522,282
                 - vec output tuple id: 22
                 - output tuple id: 22
                 - vIntermediate tuple ids: 21 
                 - hash output slot ids: 141 
                 - projections: c_customer_id
                 - project output tuple id: 22
              - BlocksProduced: sum 2.304K (2304), avg 48, max 48, min 48
              - CloseTime: avg 9.873us, max 33.519us, min 1.703us
              - ExecTime: avg 3.184ms, max 5.658ms, min 1.296ms
              - InitTime: avg 28.690us, max 474.349us, min 6.369us
              - MemoryUsage: sum 576.00 KB, avg 12.00 KB, max 12.00 KB, min 12.00 KB
              - MemoryUsagePeak: sum 1.09 MB, avg 23.25 KB, max 24.00 KB, min 12.00 KB
              - MemoryUsageProbeKeyArena: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsageProbeKeyArenaPeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 9.710us, max 196.578us, min 834ns
              - ProbeRows: sum 979.689K (979689), avg 20.41K (20410), max 20.754K (20754), min 20.128K (20128)
              - ProjectionTime: avg 601.107us, max 1.32ms, min 233.639us
              - RowsProduced: sum 172.81K (172810), avg 3.6K (3600), max 3.724K (3724), min 3.445K (3445)
              - WaitForDependency[HASH_JOIN_OPERATOR_DEPENDENCY]Time: avg 2sec349ms, max 2sec349ms, min 2sec349ms
           EXCHANGE_OPERATOR (id=19):
              - BlocksProduced: sum 2.304K (2304), avg 48, max 48, min 48
              - CloseTime: avg 3.844us, max 10.899us, min 1.257us
              - ExecTime: avg 716.355us, max 3.591ms, min 320.135us
              - InitTime: avg 28.383us, max 122.215us, min 13.704us
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 21.57 MB, avg 460.25 KB, max 612.00 KB, min 276.00 KB
              - OpenTime: avg 208ns, max 728ns, min 56ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 979.689K (979689), avg 20.41K (20410), max 20.754K (20754), min 20.128K (20128)
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForData0: avg 296.436ms, max 297.684ms, min 296.20ms
         Pipeline : 4(instance_num=48):
            - WaitWorkerTime: avg 1.353ms, max 4.128ms, min 310.778us
           HASH_JOIN_SINK_OPERATOR (id=20 , nereids_id=3102):
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 350.491us, max 563.353us, min 243.76us
              - InitTime: avg 49.815us, max 219.883us, min 11.702us
              - InputRows: sum 172.835K (172835), avg 3.6K (3600), max 3.724K (3724), min 3.445K (3445)
              - MemoryUsage: sum 2.85 MB, avg 60.74 KB, max 66.55 KB, min 49.46 KB
              - MemoryUsageBuildBlocks: sum 960.00 KB, avg 20.00 KB, max 20.00 KB, min 20.00 KB
              - MemoryUsageBuildKeyArena: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsageHashTable: sum 1.91 MB, avg 40.74 KB, max 46.55 KB, min 29.46 KB
              - MemoryUsagePeak: sum 2.85 MB, avg 60.74 KB, max 66.55 KB, min 49.46 KB
              - OpenTime: avg 31ns, max 54ns, min 20ns
              - WaitForDependency[HASH_JOIN_SINK_OPERATOR_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
           EXCHANGE_OPERATOR (id=17):
              - PlanInfo
                 - offset: 0
              - BlocksProduced: sum 2.304K (2304), avg 48, max 48, min 48
              - CloseTime: avg 9.793us, max 49.649us, min 1.554us
              - ExecTime: avg 332.951us, max 719.612us, min 220.439us
              - InitTime: avg 29.79us, max 307.728us, min 10.546us
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 131.50 KB, avg 2.74 KB, max 3.75 KB, min 1.88 KB
              - OpenTime: avg 231ns, max 955ns, min 54ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 172.835K (172835), avg 3.6K (3600), max 3.724K (3724), min 3.445K (3445)
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForData0: avg 2sec348ms, max 2sec348ms, min 2sec347ms
       Fragment 2:
         Pipeline : 0(instance_num=48):
            - WaitWorkerTime: avg 1.158ms, max 5.879ms, min 65.705us
           DATA_STREAM_SINK_OPERATOR (dest_id=19):
              - BlocksProduced: sum 2.304K (2304), avg 48, max 48, min 48
              - CloseTime: avg 8.444us, max 175.96us, min 1.659us
              - ExecTime: avg 4.719ms, max 10.923ms, min 2.616ms
              - InitTime: avg 109.515us, max 167.256us, min 66.493us
              - InputRows: sum 979.689K (979689), avg 20.41K (20410), max 24.384K (24384), min 16.818K (16818)
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 28.33 MB, avg 604.42 KB, max 884.00 KB, min 576.00 KB
              - OpenTime: avg 653.522us, max 1.393ms, min 142.287us
              - OverallThroughput: sum 0.0 /sec, avg 0.0 /sec, max 0.0 /sec, min 0.0 /sec
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForRpcBufferQueue: avg 0ns, max 0ns, min 0ns
           LOCAL_EXCHANGE_OPERATOR (PASSTHROUGH) (id=-1):
              - BlocksProduced: sum 248, avg 5, max 6, min 5
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 39.686us, max 62.696us, min 23.789us
              - GetBlockFailedTime: sum 234, avg 4, max 6, min 4
              - InitTime: avg 1.85us, max 3.663us, min 478ns
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 5.25 MB, avg 112.00 KB, max 112.00 KB, min 112.00 KB
              - OpenTime: avg 0ns, max 0ns, min 0ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 979.689K (979689), avg 20.41K (20410), max 24.384K (24384), min 16.818K (16818)
              - WaitForDependency[LOCAL_EXCHANGE_OPERATOR_DEPENDENCY]Time: avg 2sec640ms, max 2sec642ms, min 2sec638ms
         Pipeline : 1(instance_num=1):
            - WaitWorkerTime: avg 136.210us, max 136.210us, min 136.210us
           LOCAL_EXCHANGE_SINK_OPERATOR (PASSTHROUGH) (id=-1):
              - CloseTime: avg 82.510us, max 82.510us, min 82.510us
              - ExecTime: avg 1.456ms, max 1.456ms, min 1.456ms
              - InitTime: avg 2.33us, max 2.33us, min 2.33us
              - InputRows: sum 979.689K (979689), avg 979.689K (979689), max 979.689K (979689), min 979.689K (979689)
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 929ns, max 929ns, min 929ns
              - WaitForDependency[LOCAL_EXCHANGE_SINK_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
           OLAP_SCAN_OPERATOR (id=18. nereids_id=3014. table name = customer(customer)):
              - PlanInfo
                 - TABLE: tpcds.customer(customer), PREAGGREGATION: ON
                 - TOPN OPT:21
                 - runtime filters: RF006[min_max] -> c_customer_sk, RF007[in_or_bloom] -> c_customer_sk
                 - partitions=1/1 (customer)
                 - tablets=9/9, tabletList=11004,11006,11008 ...
                 - cardinality=2000000, avgRowSize=292.06763, numNodes=1
                 - pushAggOp=NONE
                 - projections: c_customer_sk, c_customer_id
                 - project output tuple id: 20
              - BlocksProduced: sum 248, avg 248, max 248, min 248
              - CloseTime: avg 111.68us, max 111.68us, min 111.68us
              - ExecTime: avg 2sec643ms, max 2sec643ms, min 2sec643ms
              - InitTime: avg 148.742us, max 148.742us, min 148.742us
              - MemoryUsage: sum 17.22 MB, avg 17.22 MB, max 17.22 MB, min 17.22 MB
              - MemoryUsagePeak: sum 23.37 MB, avg 23.37 MB, max 23.37 MB, min 23.37 MB
              - OpenTime: avg 1sec254ms, max 1sec254ms, min 1sec254ms
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 979.689K (979689), avg 979.689K (979689), max 979.689K (979689), min 979.689K (979689)
              - RuntimeFilterInfo: sum , avg , max , min 
              - WaitForDependency[OLAP_SCAN_OPERATOR_DEPENDENCY]Time: avg 420.341ms, max 420.341ms, min 420.341ms
       Fragment 3:
         Pipeline : 0(instance_num=48):
            - WaitWorkerTime: avg 923.879us, max 10.700ms, min 25.443us
           DATA_STREAM_SINK_OPERATOR (dest_id=17):
              - BlocksProduced: sum 2.304K (2304), avg 48, max 48, min 48
              - CloseTime: avg 2.402us, max 5.502us, min 1.215us
              - ExecTime: avg 4.970ms, max 8.519ms, min 2.473ms
              - InitTime: avg 151.181us, max 888.969us, min 75.679us
              - InputRows: sum 172.835K (172835), avg 3.6K (3600), max 3.963K (3963), min 3.296K (3296)
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 1.40 MB, avg 29.88 KB, max 30.25 KB, min 29.25 KB
              - OpenTime: avg 507.482us, max 1.166ms, min 197.566us
              - OverallThroughput: sum 0.0 /sec, avg 0.0 /sec, max 0.0 /sec, min 0.0 /sec
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForRpcBufferQueue: avg 0ns, max 0ns, min 0ns
           HASH_JOIN_OPERATOR (id=16 , nereids_id=3087):
              - PlanInfo
                 - join op: INNER JOIN(BROADCAST)[]
                 - equal join conjunct: (ctr_store_sk = ctr_store_sk)
                 - other join predicates: (CAST(ctr_total_return AS double) > CAST((avg(cast(ctr_total_return as DECIMALV3(38, 4))) * 1.2) AS double))
                 - cardinality=516,012
                 - vec output tuple id: 18
                 - output tuple id: 18
                 - vIntermediate tuple ids: 17 
                 - hash output slot ids: 113 115 75 
                 - projections: ctr_customer_sk
                 - project output tuple id: 18
              - BlocksProduced: sum 1.345K (1345), avg 28, max 30, min 26
              - CloseTime: avg 21.17us, max 43.47us, min 5.699us
              - ExecTime: avg 11.414ms, max 14.358ms, min 4.377ms
              - InitTime: avg 24.652us, max 69.484us, min 10.973us
              - MemoryUsage: sum 890.00 KB, avg 18.54 KB, max 27.00 KB, min 13.50 KB
              - MemoryUsagePeak: sum 1.27 MB, avg 27.00 KB, max 27.00 KB, min 27.00 KB
              - MemoryUsageProbeKeyArena: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsageProbeKeyArenaPeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 5.861us, max 108.365us, min 917ns
              - ProbeRows: sum 671.479K (671479), avg 13.989K (13989), max 15.13K (15130), min 12.806K (12806)
              - ProjectionTime: avg 145.732us, max 383.884us, min 55.415us
              - RowsProduced: sum 172.835K (172835), avg 3.6K (3600), max 3.963K (3963), min 3.296K (3296)
              - WaitForDependency[HASH_JOIN_OPERATOR_DEPENDENCY]Time: avg 2sec324ms, max 2sec325ms, min 2sec324ms
           LOCAL_EXCHANGE_OPERATOR (PASSTHROUGH) (id=-6):
              - BlocksProduced: sum 1.345K (1345), avg 28, max 30, min 26
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 672.538us, max 940.810us, min 309.752us
              - GetBlockFailedTime: sum 0, avg 0, max 0, min 0
              - InitTime: avg 1.827us, max 36.92us, min 488ns
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 25.00 MB, avg 533.24 KB, max 603.50 KB, min 450.50 KB
              - OpenTime: avg 0ns, max 0ns, min 0ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 671.479K (671479), avg 13.989K (13989), max 15.13K (15130), min 12.806K (12806)
              - WaitForDependency[LOCAL_EXCHANGE_OPERATOR_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
         Pipeline : 1(instance_num=48):
            - WaitWorkerTime: avg 3.467ms, max 7.929ms, min 1.401ms
           LOCAL_EXCHANGE_SINK_OPERATOR (PASSTHROUGH) (id=-6):
              - CloseTime: avg 1.20us, max 14.633us, min 183ns
              - ExecTime: avg 187.359us, max 253.390us, min 137.119us
              - InitTime: avg 3.612us, max 34.262us, min 930ns
              - InputRows: sum 671.479K (671479), avg 13.989K (13989), max 15.085K (15085), min 12.718K (12718)
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 130ns, max 596ns, min 38ns
              - WaitForDependency[LOCAL_EXCHANGE_SINK_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
           HASH_JOIN_OPERATOR (id=15 , nereids_id=3051):
              - PlanInfo
                 - join op: INNER JOIN(BROADCAST)[]
                 - equal join conjunct: (ctr_store_sk = s_store_sk)
                 - runtime filters: RF002[min_max] <- s_store_sk(40/64/1048576), RF003[in_or_bloom] <- s_store_sk(40/64/1048576)
                 - cardinality=1,032,024
                 - vec output tuple id: 16
                 - output tuple id: 16
                 - vIntermediate tuple ids: 15 
                 - hash output slot ids: 106 107 108 
                 - projections: ctr_customer_sk, ctr_store_sk, ctr_total_return
                 - project output tuple id: 16
              - BlocksProduced: sum 1.345K (1345), avg 28, max 30, min 26
              - CloseTime: avg 14.506us, max 41.677us, min 5.923us
              - ExecTime: avg 4.598ms, max 13.177ms, min 2.148ms
              - InitTime: avg 13.613us, max 48.166us, min 4.484us
              - MemoryUsage: sum 888.50 KB, avg 18.51 KB, max 27.00 KB, min 13.50 KB
              - MemoryUsagePeak: sum 1.27 MB, avg 27.00 KB, max 27.00 KB, min 27.00 KB
              - MemoryUsageProbeKeyArena: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsageProbeKeyArenaPeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 5.444us, max 51.775us, min 1.89us
              - ProbeRows: sum 671.479K (671479), avg 13.989K (13989), max 15.085K (15085), min 12.718K (12718)
              - ProjectionTime: avg 1.189ms, max 5.645ms, min 499.623us
              - RowsProduced: sum 671.479K (671479), avg 13.989K (13989), max 15.085K (15085), min 12.718K (12718)
              - WaitForDependency[HASH_JOIN_OPERATOR_DEPENDENCY]Time: avg 41.679ms, max 42.222ms, min 41.350ms
           LOCAL_EXCHANGE_OPERATOR (PASSTHROUGH) (id=-5):
              - BlocksProduced: sum 1.345K (1345), avg 28, max 30, min 26
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 345.698us, max 5.142ms, min 128.282us
              - GetBlockFailedTime: sum 1.102K (1102), avg 22, max 28, min 17
              - InitTime: avg 1.510us, max 31.112us, min 397ns
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 2.40 MB, avg 51.24 KB, max 126.00 KB, min 27.00 KB
              - OpenTime: avg 0ns, max 0ns, min 0ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 671.479K (671479), avg 13.989K (13989), max 15.085K (15085), min 12.718K (12718)
              - WaitForDependency[LOCAL_EXCHANGE_OPERATOR_DEPENDENCY]Time: avg 2sec272ms, max 2sec276ms, min 2sec264ms
         Pipeline : 2(instance_num=48):
            - WaitWorkerTime: avg 4.921ms, max 8.109ms, min 2.8ms
           LOCAL_EXCHANGE_SINK_OPERATOR (PASSTHROUGH) (id=-5):
              - CloseTime: avg 2.119us, max 76.455us, min 101ns
              - ExecTime: avg 196.149us, max 356.957us, min 125.171us
              - InitTime: avg 3.812us, max 34.856us, min 794ns
              - InputRows: sum 671.479K (671479), avg 13.989K (13989), max 17.469K (17469), min 10.447K (10447)
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 101ns, max 296ns, min 33ns
              - WaitForDependency[LOCAL_EXCHANGE_SINK_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
           EXCHANGE_OPERATOR (id=14):
              - PlanInfo
                 - offset: 0
              - BlocksProduced: sum 1.345K (1345), avg 28, max 35, min 21
              - CloseTime: avg 27.614us, max 221.140us, min 2.828us
              - ExecTime: avg 419.61us, max 633.618us, min 247.72us
              - InitTime: avg 36.68us, max 186.789us, min 9.865us
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 2.89 MB, avg 61.75 KB, max 151.50 KB, min 27.00 KB
              - OpenTime: avg 168ns, max 568ns, min 36ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 671.479K (671479), avg 13.989K (13989), max 17.469K (17469), min 10.447K (10447)
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForData0: avg 2sec315ms, max 2sec317ms, min 2sec310ms
         Pipeline : 3(instance_num=48):
            - WaitWorkerTime: avg 336.387us, max 813.598us, min 31.620us
           HASH_JOIN_SINK_OPERATOR (id=16 , nereids_id=3087):
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 93.49us, max 237.64us, min 11.925us
              - InitTime: avg 19.282us, max 57.877us, min 5.207us
              - InputRows: sum 202, avg 4, max 202, min 0
              - MemoryUsage: sum 7.30 KB, avg 155.00 B, max 7.30 KB, min 0.00 
              - MemoryUsageBuildBlocks: sum 5.50 KB, avg 117.00 B, max 5.50 KB, min 0.00 
              - MemoryUsageBuildKeyArena: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsageHashTable: sum 1.80 KB, avg 38.00 B, max 1.80 KB, min 0.00 
              - MemoryUsagePeak: sum 7.30 KB, avg 155.00 B, max 7.30 KB, min 0.00 
              - OpenTime: avg 28ns, max 48ns, min 20ns
              - WaitForDependency[HASH_JOIN_SINK_OPERATOR_DEPENDENCY]Time: avg 2sec276ms, max 2sec325ms, min 0ns
           EXCHANGE_OPERATOR (id=11):
              - PlanInfo
                 - offset: 0
              - BlocksProduced: sum 47, avg 0, max 47, min 0
              - CloseTime: avg 10.830us, max 31.637us, min 2.634us
              - ExecTime: avg 94.369us, max 1.152ms, min 18.33us
              - InitTime: avg 71.655us, max 1.147ms, min 10.959us
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 130.13 KB, avg 2.71 KB, max 130.13 KB, min 0.00 
              - OpenTime: avg 204ns, max 518ns, min 52ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 202, avg 4, max 202, min 0
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
         Pipeline : 4(instance_num=48):
            - WaitWorkerTime: avg 366.364us, max 1.17ms, min 33.926us
           HASH_JOIN_SINK_OPERATOR (id=15 , nereids_id=3051):
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 152.635us, max 906.833us, min 21.12us
              - InitTime: avg 79.51us, max 804.875us, min 11.289us
              - InputRows: sum 45, avg 0, max 45, min 0
              - MemoryUsage: sum 700.00 B, avg 14.00 B, max 700.00 B, min 0.00 
              - MemoryUsageBuildBlocks: sum 256.00 B, avg 5.00 B, max 256.00 B, min 0.00 
              - MemoryUsageBuildKeyArena: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsageHashTable: sum 444.00 B, avg 9.00 B, max 444.00 B, min 0.00 
              - MemoryUsagePeak: sum 700.00 B, avg 14.00 B, max 700.00 B, min 0.00 
              - OpenTime: avg 30ns, max 58ns, min 20ns
              - WaitForDependency[HASH_JOIN_SINK_OPERATOR_DEPENDENCY]Time: avg 40.683ms, max 42.36ms, min 0ns
           EXCHANGE_OPERATOR (id=13):
              - PlanInfo
                 - offset: 0
              - BlocksProduced: sum 1, avg 0, max 1, min 0
              - CloseTime: avg 17.998us, max 76.55us, min 4.873us
              - ExecTime: avg 75.51us, max 777.571us, min 16.136us
              - InitTime: avg 55.719us, max 770.156us, min 8.713us
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 256.00 B, avg 5.00 B, max 256.00 B, min 0.00 
              - OpenTime: avg 246ns, max 1.395us, min 99ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 45, avg 0, max 45, min 0
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForData0: avg 40.959ms, max 40.959ms, min 40.959ms
       Fragment 4:
         Pipeline : 0(instance_num=48):
            - WaitWorkerTime: avg 132.387us, max 847.458us, min 13.817us
           DATA_STREAM_SINK_OPERATOR (dest_id=13):
              - BlocksProduced: sum 49, avg 1, max 2, min 1
              - CloseTime: avg 528ns, max 1.155us, min 203ns
              - ExecTime: avg 37.922us, max 86.307us, min 16.925us
              - InitTime: avg 18.947us, max 45.640us, min 9.639us
              - InputRows: sum 45, avg 0, max 45, min 0
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 3.942us, max 21.552us, min 916ns
              - OverallThroughput: sum 0.0 /sec, avg 0.0 /sec, max 0.0 /sec, min 0.0 /sec
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForRpcBufferQueue: avg 0ns, max 0ns, min 0ns
           LOCAL_EXCHANGE_OPERATOR (PASSTHROUGH) (id=-1):
              - BlocksProduced: sum 1, avg 0, max 1, min 0
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 2.214us, max 25.390us, min 842ns
              - GetBlockFailedTime: sum 0, avg 0, max 0, min 0
              - InitTime: avg 963ns, max 3.878us, min 435ns
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 256.00 B, avg 5.00 B, max 256.00 B, min 0.00 
              - OpenTime: avg 0ns, max 0ns, min 0ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 45, avg 0, max 45, min 0
              - WaitForDependency[LOCAL_EXCHANGE_OPERATOR_DEPENDENCY]Time: avg 40.0ms, max 40.455ms, min 39.736ms
         Pipeline : 1(instance_num=1):
            - WaitWorkerTime: avg 224.904us, max 224.904us, min 224.904us
           LOCAL_EXCHANGE_SINK_OPERATOR (PASSTHROUGH) (id=-1):
              - CloseTime: avg 133.50us, max 133.50us, min 133.50us
              - ExecTime: avg 146.641us, max 146.641us, min 146.641us
              - InitTime: avg 1.921us, max 1.921us, min 1.921us
              - InputRows: sum 45, avg 45, max 45, min 45
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 726ns, max 726ns, min 726ns
              - WaitForDependency[LOCAL_EXCHANGE_SINK_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
           OLAP_SCAN_OPERATOR (id=12. nereids_id=3031. table name = store(store)):
              - PlanInfo
                 - TABLE: tpcds.store(store), PREAGGREGATION: ON
                 - PREDICATES: (s_state = 'TN')
                 - partitions=1/1 (store)
                 - tablets=1/1, tabletList=10738
                 - cardinality=402, avgRowSize=693.3334, numNodes=1
                 - pushAggOp=NONE
                 - projections: s_store_sk
                 - project output tuple id: 13
              - BlocksProduced: sum 1, avg 1, max 1, min 1
              - CloseTime: avg 33.459us, max 33.459us, min 33.459us
              - ExecTime: avg 40.290ms, max 40.290ms, min 40.290ms
              - InitTime: avg 84.299us, max 84.299us, min 84.299us
              - MemoryUsage: sum 256.00 B, avg 256.00 B, max 256.00 B, min 256.00 B
              - MemoryUsagePeak: sum 256.00 B, avg 256.00 B, max 256.00 B, min 256.00 B
              - OpenTime: avg 23.652ms, max 23.652ms, min 23.652ms
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 45, avg 45, max 45, min 45
              - RuntimeFilterInfo: sum , avg , max , min 
              - WaitForDependency[OLAP_SCAN_OPERATOR_DEPENDENCY]Time: avg 16.381ms, max 16.381ms, min 16.381ms
       Fragment 5:
         Pipeline : 0(instance_num=48):
            - WaitWorkerTime: avg 18.73us, max 43.879us, min 8.990us
           DATA_STREAM_SINK_OPERATOR (dest_id=11):
              - BlocksProduced: sum 48, avg 1, max 1, min 1
              - CloseTime: avg 474ns, max 889ns, min 191ns
              - ExecTime: avg 31.10us, max 55.321us, min 13.638us
              - InitTime: avg 19.153us, max 35.928us, min 8.69us
              - InputRows: sum 202, avg 4, max 9, min 0
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 1.694us, max 7.547us, min 838ns
              - OverallThroughput: sum 0.0 /sec, avg 0.0 /sec, max 0.0 /sec, min 0.0 /sec
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForRpcBufferQueue: avg 0ns, max 0ns, min 0ns
           AGGREGATION_OPERATOR (id=10 , nereids_id=3077):
              - PlanInfo
                 - output: avg(partial_avg(cast(ctr_total_return as DECIMALV3(38, 4))))[#75]
                 - group by: ctr_store_sk
                 - sortByGroupKey:false
                 - cardinality=201
              - BlocksProduced: sum 47, avg 0, max 1, min 0
              - CloseTime: avg 1.129us, max 6.878us, min 429ns
              - ExecTime: avg 29.949us, max 108.128us, min 15.952us
              - InitTime: avg 8.891us, max 28.43us, min 3.497us
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 0ns, max 0ns, min 0ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 202, avg 4, max 9, min 0
              - WaitForDependency[AGGREGATION_OPERATOR_DEPENDENCY]Time: avg 2sec323ms, max 2sec323ms, min 2sec322ms
         Pipeline : 1(instance_num=48):
            - WaitWorkerTime: avg 1.205ms, max 2.35ms, min 62.645us
           AGGREGATION_SINK_OPERATOR (id=10 , nereids_id=3077):
              - CloseTime: avg 1.322us, max 8.150us, min 462ns
              - ExecTime: avg 365.72us, max 1.811ms, min 39.911us
              - InitTime: avg 8.779us, max 31.38us, min 4.527us
              - InputRows: sum 9.696K (9696), avg 202, max 432, min 0
              - MemoryUsage: sum 19.65 MB, avg 419.25 KB, max 432.11 KB, min 0.00 
              - MemoryUsageHashTable: sum 4.23 KB, avg 90.00 B, max 240.00 B, min 0.00 
              - MemoryUsagePeak: sum 19.65 MB, avg 419.25 KB, max 432.11 KB, min 0.00 
              - MemoryUsageSerializeKeyArena: sum 19.65 MB, avg 419.17 KB, max 432.00 KB, min 0.00 
              - OpenTime: avg 24.647us, max 231.943us, min 2.263us
              - WaitForDependency[AGGREGATION_SINK_OPERATOR_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
           EXCHANGE_OPERATOR (id=9):
              - PlanInfo
                 - offset: 0
              - BlocksProduced: sum 2.256K (2256), avg 47, max 48, min 0
              - CloseTime: avg 9.519us, max 55.681us, min 2.554us
              - ExecTime: avg 517.472us, max 1.179ms, min 43.470us
              - InitTime: avg 66.0us, max 196.544us, min 14.829us
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 241.69 KB, avg 5.04 KB, max 16.25 KB, min 0.00 
              - OpenTime: avg 190ns, max 534ns, min 35ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 9.696K (9696), avg 202, max 432, min 0
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForData0: avg 2sec320ms, max 2sec322ms, min 2sec320ms
       Fragment 6:
         Pipeline : 0(instance_num=48):
            - WaitWorkerTime: avg 4.142ms, max 9.703ms, min 1.0ms
           DATA_STREAM_SINK_OPERATOR (dest_id=9):
              - BlocksProduced: sum 2.304K (2304), avg 48, max 48, min 48
              - CloseTime: avg 2.882us, max 8.418us, min 1.332us
              - ExecTime: avg 4.134ms, max 5.332ms, min 1.839ms
              - InitTime: avg 138.748us, max 286.317us, min 66.595us
              - InputRows: sum 9.696K (9696), avg 202, max 202, min 202
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 630.522us, max 1.247ms, min 76.883us
              - OverallThroughput: sum 0.0 /sec, avg 0.0 /sec, max 0.0 /sec, min 0.0 /sec
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForRpcBufferQueue: avg 0ns, max 0ns, min 0ns
           STREAMING_AGGREGATION_OPERATOR (id=8 , nereids_id=3067):
              - PlanInfo
                 - STREAMING
                 - output: partial_avg(CAST(ctr_total_return AS decimalv3(38,4)))[#73]
                 - group by: ctr_store_sk
                 - sortByGroupKey:false
                 - cardinality=201
              - BlocksProduced: sum 48, avg 1, max 1, min 1
              - CloseTime: avg 7.528us, max 16.420us, min 3.274us
              - ExecTime: avg 6.202ms, max 19.764ms, min 3.876ms
              - InitTime: avg 14.61us, max 70.776us, min 7.364us
              - MemoryUsage: sum 20.44 MB, avg 435.98 KB, max 435.98 KB, min 435.98 KB
              - MemoryUsageHashTable: sum 191.25 KB, avg 3.98 KB, max 3.98 KB, min 3.98 KB
              - MemoryUsagePeak: sum 20.44 MB, avg 435.98 KB, max 435.98 KB, min 435.98 KB
              - MemoryUsageSerializeKeyArena: sum 20.25 MB, avg 432.00 KB, max 432.00 KB, min 432.00 KB
              - MemoryUsageSerializeKeyArenaPeak: sum 20.25 MB, avg 432.00 KB, max 432.00 KB, min 432.00 KB
              - OpenTime: avg 35.257us, max 262.425us, min 2.954us
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 9.696K (9696), avg 202, max 202, min 202
           EXCHANGE_OPERATOR (id=7):
              - BlocksProduced: sum 1.345K (1345), avg 28, max 34, min 19
              - CloseTime: avg 27.180us, max 238.736us, min 2.81us
              - ExecTime: avg 859.748us, max 5.623ms, min 218.955us
              - InitTime: avg 30.384us, max 86.23us, min 10.658us
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 15.98 MB, avg 341.00 KB, max 704.00 KB, min 88.00 KB
              - OpenTime: avg 309ns, max 2.190us, min 37ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 5.435529M (5435529), avg 113.24K (113240), max 137.658K (137658), min 75.846K (75846)
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForData0: avg 2sec306ms, max 2sec311ms, min 2sec295ms
       Fragment 7:
         Pipeline : 0(instance_num=48):
            - WaitWorkerTime: avg 546.79us, max 5.458ms, min 21.66us
           MULTI_CAST_DATA_STREAM_SINK_OPERATOR (dest_id=7,14,):
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 362.931us, max 5.514ms, min 47.883us
              - InitTime: avg 0ns, max 0ns, min 0ns
              - InputRows: sum 10.871058M (10871058), avg 226.48K (226480), max 227.774K (227774), min 224.734K (224734)
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 0ns, max 0ns, min 0ns
              - WaitForDependency[MULTI_CAST_DATA_STREAM_SINK_OPERATOR_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
           AGGREGATION_OPERATOR (id=6 , nereids_id=3003):
              - PlanInfo
                 - output: sum(partial_sum(sr_return_amt))[#66]
                 - group by: sr_customer_sk, sr_store_sk
                 - sortByGroupKey:false
                 - cardinality=5,160,122
                 - projections: sr_customer_sk, sr_store_sk, ctr_total_return
                 - project output tuple id: 8
              - BlocksProduced: sum 1.345K (1345), avg 28, max 29, min 28
              - CloseTime: avg 892ns, max 1.704us, min 414ns
              - ExecTime: avg 12.750ms, max 35.468ms, min 5.303ms
              - InitTime: avg 9.580us, max 51.100us, min 3.794us
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 0ns, max 0ns, min 0ns
              - ProjectionTime: avg 5.714ms, max 28.113ms, min 1.131ms
              - RowsProduced: sum 5.435529M (5435529), avg 113.24K (113240), max 113.887K (113887), min 112.367K (112367)
              - WaitForDependency[AGGREGATION_OPERATOR_DEPENDENCY]Time: avg 2sec283ms, max 2sec309ms, min 2sec253ms
         Pipeline : 1(instance_num=48):
            - WaitWorkerTime: avg 3.857ms, max 9.172ms, min 21.405us
           AGGREGATION_SINK_OPERATOR (id=6 , nereids_id=3003):
              - CloseTime: avg 3.811us, max 12.311us, min 1.172us
              - ExecTime: avg 84.500ms, max 109.276ms, min 53.385ms
              - InitTime: avg 12.441us, max 70.97us, min 4.525us
              - InputRows: sum 5.539497M (5539497), avg 115.406K (115406), max 140.574K (140574), min 113.975K (113975)
              - MemoryUsage: sum 534.00 MB, avg 11.12 MB, max 11.12 MB, min 11.12 MB
              - MemoryUsageHashTable: sum 144.00 MB, avg 3.00 MB, max 3.00 MB, min 3.00 MB
              - MemoryUsagePeak: sum 534.00 MB, avg 11.12 MB, max 11.12 MB, min 11.12 MB
              - MemoryUsageSerializeKeyArena: sum 390.00 MB, avg 8.13 MB, max 8.13 MB, min 8.13 MB
              - OpenTime: avg 135.620us, max 1.449ms, min 3.684us
              - WaitForDependency[AGGREGATION_SINK_OPERATOR_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
           EXCHANGE_OPERATOR (id=5):
              - PlanInfo
                 - offset: 0
              - BlocksProduced: sum 2.304K (2304), avg 48, max 48, min 48
              - CloseTime: avg 5.560us, max 8.870us, min 3.295us
              - ExecTime: avg 942.577us, max 1.732ms, min 657.798us
              - InitTime: avg 33.240us, max 90.164us, min 13.758us
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 166.05 MB, avg 3.46 MB, max 4.10 MB, min 2.70 MB
              - OpenTime: avg 316ns, max 1.842us, min 45ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 5.539497M (5539497), avg 115.406K (115406), max 140.574K (140574), min 113.975K (113975)
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForData0: avg 2sec194ms, max 2sec195ms, min 2sec192ms
         Pipeline : 2(instance_num=48):
            - WaitWorkerTime: avg 2.44ms, max 6.270ms, min 408.855us
           DATA_STREAM_SINK_OPERATOR (dest_id=7):
              - BlocksProduced: sum 3.601K (3601), avg 75, max 76, min 75
              - CloseTime: avg 2.385us, max 3.777us, min 1.385us
              - ExecTime: avg 1.122ms, max 2.469ms, min 629.889us
              - InitTime: avg 158.924us, max 305.158us, min 65.332us
              - InputRows: sum 5.435529M (5435529), avg 113.24K (113240), max 113.887K (113887), min 112.367K (112367)
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 694.20us, max 1.66ms, min 298.809us
              - OverallThroughput: sum 0.0 /sec, avg 0.0 /sec, max 0.0 /sec, min 0.0 /sec
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForRpcBufferQueue: avg 0ns, max 0ns, min 0ns
           MULTI_CAST_DATA_STREAM_SOURCE_OPERATOR (id=-1):
              - BlocksProduced: sum 1.345K (1345), avg 28, max 29, min 28
              - CloseTime: avg 1.104us, max 2.863us, min 429ns
              - ExecTime: avg 4.351ms, max 13.672ms, min 1.398ms
              - InitTime: avg 5.783us, max 66.703us, min 1.774us
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 4.262us, max 32.555us, min 752ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 5.435529M (5435529), avg 113.24K (113240), max 113.887K (113887), min 112.367K (112367)
              - WaitForDependency[MULTI_CAST_DATA_STREAM_SOURCE_OPERATOR_DEPENDENCY]Time: avg 2sec290ms, max 2sec312ms, min 2sec253ms
         Pipeline : 3(instance_num=48):
            - WaitWorkerTime: avg 1.430ms, max 4.219ms, min 125.4us
           DATA_STREAM_SINK_OPERATOR (dest_id=14):
              - BlocksProduced: sum 3.601K (3601), avg 75, max 76, min 75
              - CloseTime: avg 3.728us, max 27.197us, min 1.190us
              - ExecTime: avg 641.301us, max 838.918us, min 413.884us
              - InitTime: avg 167.849us, max 324.144us, min 65.164us
              - InputRows: sum 671.479K (671479), avg 13.989K (13989), max 14.193K (14193), min 13.717K (13717)
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 212.372us, max 324.410us, min 50.519us
              - OverallThroughput: sum 0.0 /sec, avg 0.0 /sec, max 0.0 /sec, min 0.0 /sec
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForRpcBufferQueue: avg 0ns, max 0ns, min 0ns
           MULTI_CAST_DATA_STREAM_SOURCE_OPERATOR (id=-1):
              - BlocksProduced: sum 1.345K (1345), avg 28, max 29, min 28
              - CloseTime: avg 1.76us, max 2.348us, min 348ns
              - ExecTime: avg 8.169ms, max 21.136ms, min 3.15ms
              - InitTime: avg 21.287us, max 87.580us, min 5.574us
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 109.572us, max 200.291us, min 35.141us
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 671.479K (671479), avg 13.989K (13989), max 14.193K (14193), min 13.717K (13717)
              - WaitForDependency[MULTI_CAST_DATA_STREAM_SOURCE_OPERATOR_DEPENDENCY]Time: avg 2sec248ms, max 2sec270ms, min 2sec212ms
       Fragment 8:
         Pipeline : 0(instance_num=48):
            - WaitWorkerTime: avg 18.976ms, max 39.249ms, min 3.174ms
           DATA_STREAM_SINK_OPERATOR (dest_id=5):
              - BlocksProduced: sum 2.304K (2304), avg 48, max 48, min 48
              - CloseTime: avg 3.133us, max 26.566us, min 1.355us
              - ExecTime: avg 29.49ms, max 50.61ms, min 16.842ms
              - InitTime: avg 142.971us, max 249.628us, min 84.191us
              - InputRows: sum 5.539497M (5539497), avg 115.406K (115406), max 117.069K (117069), min 111.492K (111492)
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 270.00 MB, avg 5.63 MB, max 5.63 MB, min 5.63 MB
              - OpenTime: avg 560.256us, max 872.890us, min 157.36us
              - OverallThroughput: sum 0.0 /sec, avg 0.0 /sec, max 0.0 /sec, min 0.0 /sec
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForRpcBufferQueue: avg 0ns, max 0ns, min 0ns
           STREAMING_AGGREGATION_OPERATOR (id=4 , nereids_id=2993):
              - PlanInfo
                 - STREAMING
                 - output: partial_sum(sr_return_amt)[#63]
                 - group by: sr_customer_sk, sr_store_sk
                 - sortByGroupKey:false
                 - cardinality=5,160,122
              - BlocksProduced: sum 1.376K (1376), avg 28, max 29, min 28
              - CloseTime: avg 473.855us, max 761.765us, min 329.397us
              - ExecTime: avg 18.475ms, max 22.832ms, min 14.348ms
              - InitTime: avg 14.864us, max 65.463us, min 7.496us
              - MemoryUsage: sum 264.00 MB, avg 5.50 MB, max 5.50 MB, min 5.50 MB
              - MemoryUsageHashTable: sum 72.00 MB, avg 1.50 MB, max 1.50 MB, min 1.50 MB
              - MemoryUsagePeak: sum 264.00 MB, avg 5.50 MB, max 5.50 MB, min 5.50 MB
              - MemoryUsageSerializeKeyArena: sum 192.00 MB, avg 4.00 MB, max 4.00 MB, min 4.00 MB
              - MemoryUsageSerializeKeyArenaPeak: sum 192.00 MB, avg 4.00 MB, max 4.00 MB, min 4.00 MB
              - OpenTime: avg 22.173us, max 77.567us, min 5.625us
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 5.539497M (5539497), avg 115.406K (115406), max 117.069K (117069), min 111.492K (111492)
           HASH_JOIN_OPERATOR (id=3 , nereids_id=2983):
              - BlocksProduced: sum 1.377K (1377), avg 28, max 29, min 28
              - CloseTime: avg 13.394us, max 23.575us, min 8.346us
              - ExecTime: avg 3.378ms, max 3.985ms, min 2.834ms
              - InitTime: avg 22.705us, max 137.312us, min 7.91us
              - MemoryUsage: sum 3.75 MB, avg 80.00 KB, max 80.00 KB, min 80.00 KB
              - MemoryUsagePeak: sum 3.75 MB, avg 80.00 KB, max 80.00 KB, min 80.00 KB
              - MemoryUsageProbeKeyArena: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsageProbeKeyArenaPeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 6.778us, max 21.898us, min 1.437us
              - ProbeRows: sum 5.580151M (5580151), avg 116.253K (116253), max 117.856K (117856), min 112.295K (112295)
              - ProjectionTime: avg 475.132us, max 556.858us, min 372.535us
              - RowsProduced: sum 5.580151M (5580151), avg 116.253K (116253), max 117.856K (117856), min 112.295K (112295)
              - WaitForDependency[HASH_JOIN_OPERATOR_DEPENDENCY]Time: avg 195.144ms, max 195.520ms, min 194.977ms
           LOCAL_EXCHANGE_OPERATOR (PASSTHROUGH) (id=-5):
              - BlocksProduced: sum 1.377K (1377), avg 28, max 29, min 28
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 172.92us, max 242.680us, min 120.137us
              - GetBlockFailedTime: sum 939, avg 19, max 21, min 17
              - InitTime: avg 1.291us, max 7.968us, min 552ns
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 7.34 MB, avg 156.67 KB, max 160.00 KB, min 80.00 KB
              - OpenTime: avg 0ns, max 0ns, min 0ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 5.580151M (5580151), avg 116.253K (116253), max 117.856K (117856), min 112.295K (112295)
              - WaitForDependency[LOCAL_EXCHANGE_OPERATOR_DEPENDENCY]Time: avg 1sec944ms, max 1sec955ms, min 1sec936ms
         Pipeline : 1(instance_num=1):
            - WaitWorkerTime: avg 1.75ms, max 1.75ms, min 1.75ms
           LOCAL_EXCHANGE_SINK_OPERATOR (PASSTHROUGH) (id=-5):
              - CloseTime: avg 11.213us, max 11.213us, min 11.213us
              - ExecTime: avg 4.950ms, max 4.950ms, min 4.950ms
              - InitTime: avg 1.878us, max 1.878us, min 1.878us
              - InputRows: sum 5.580151M (5580151), avg 5.580151M (5580151), max 5.580151M (5580151), min 5.580151M (5580151)
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 1.250us, max 1.250us, min 1.250us
              - WaitForDependency[LOCAL_EXCHANGE_SINK_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
           OLAP_SCAN_OPERATOR (id=2. nereids_id=2957. table name = store_returns(store_returns)):
              - PlanInfo
                 - TABLE: tpcds.store_returns(store_returns), PREAGGREGATION: ON
                 - runtime filters: RF000[min_max] -> sr_returned_date_sk, RF001[in_or_bloom] -> sr_returned_date_sk
                 - partitions=1/1 (store_returns)
                 - tablets=16/16, tabletList=10769,10771,10773 ...
                 - cardinality=28795080, avgRowSize=198.15651, numNodes=1
                 - pushAggOp=NONE
                 - projections: sr_returned_date_sk, sr_customer_sk, sr_store_sk, sr_return_amt
                 - project output tuple id: 3
              - BlocksProduced: sum 1.377K (1377), avg 1.377K (1377), max 1.377K (1377), min 1.377K (1377)
              - CloseTime: avg 355.689us, max 355.689us, min 355.689us
              - ExecTime: avg 2sec177ms, max 2sec177ms, min 2sec177ms
              - InitTime: avg 115.528us, max 115.528us, min 115.528us
              - MemoryUsage: sum 33.68 MB, avg 33.68 MB, max 33.68 MB, min 33.68 MB
              - MemoryUsagePeak: sum 33.68 MB, avg 33.68 MB, max 33.68 MB, min 33.68 MB
              - OpenTime: avg 223.332ms, max 223.332ms, min 223.332ms
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 5.580151M (5580151), avg 5.580151M (5580151), max 5.580151M (5580151), min 5.580151M (5580151)
              - RuntimeFilterInfo: sum , avg , max , min 
              - WaitForDependency[OLAP_SCAN_OPERATOR_DEPENDENCY]Time: avg 1sec741ms, max 1sec741ms, min 1sec741ms
         Pipeline : 2(instance_num=48):
            - WaitWorkerTime: avg 575.126us, max 1.92ms, min 27.3us
           HASH_JOIN_SINK_OPERATOR (id=3 , nereids_id=2983):
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 142.41us, max 363.684us, min 24.776us
              - InitTime: avg 33.716us, max 197.970us, min 12.950us
              - InputRows: sum 366, avg 7, max 366, min 0
              - MemoryUsage: sum 5.44 KB, avg 116.00 B, max 5.44 KB, min 0.00 
              - MemoryUsageBuildBlocks: sum 2.00 KB, avg 42.00 B, max 2.00 KB, min 0.00 
              - MemoryUsageBuildKeyArena: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsageHashTable: sum 3.44 KB, avg 73.00 B, max 3.44 KB, min 0.00 
              - MemoryUsagePeak: sum 5.44 KB, avg 116.00 B, max 5.44 KB, min 0.00 
              - OpenTime: avg 31ns, max 52ns, min 19ns
              - WaitForDependency[HASH_JOIN_SINK_OPERATOR_DEPENDENCY]Time: avg 191.415ms, max 196.39ms, min 0ns
           LOCAL_EXCHANGE_OPERATOR (PASS_TO_ONE) (id=-4):
              - BlocksProduced: sum 9, avg 0, max 9, min 0
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 2.361us, max 54.27us, min 569ns
              - GetBlockFailedTime: sum 6, avg 0, max 6, min 0
              - InitTime: avg 1.211us, max 6.764us, min 487ns
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 512.00 B, avg 10.00 B, max 512.00 B, min 0.00 
              - OpenTime: avg 0ns, max 0ns, min 0ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 366, avg 7, max 366, min 0
              - WaitForDependency[LOCAL_EXCHANGE_OPERATOR_DEPENDENCY]Time: avg 4.76ms, max 195.659ms, min 0ns
         Pipeline : 3(instance_num=1):
            - WaitWorkerTime: avg 664.105us, max 664.105us, min 664.105us
           LOCAL_EXCHANGE_SINK_OPERATOR (PASS_TO_ONE) (id=-4):
              - CloseTime: avg 6.968us, max 6.968us, min 6.968us
              - ExecTime: avg 50.217us, max 50.217us, min 50.217us
              - InitTime: avg 5.833us, max 5.833us, min 5.833us
              - InputRows: sum 366, avg 366, max 366, min 366
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 1.212us, max 1.212us, min 1.212us
              - WaitForDependency[LOCAL_EXCHANGE_SINK_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
           EXCHANGE_OPERATOR (id=1):
              - PlanInfo
                 - offset: 0
              - BlocksProduced: sum 9, avg 9, max 9, min 9
              - CloseTime: avg 7.65us, max 7.65us, min 7.65us
              - ExecTime: avg 75.415us, max 75.415us, min 75.415us
              - InitTime: avg 15.845us, max 15.845us, min 15.845us
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 512.00 B, avg 512.00 B, max 512.00 B, min 512.00 B
              - OpenTime: avg 313ns, max 313ns, min 313ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 366, avg 366, max 366, min 366
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForData0: avg 195.102ms, max 195.102ms, min 195.102ms
       Fragment 9:
         Pipeline : 0(instance_num=48):
            - WaitWorkerTime: avg 547.15us, max 834.789us, min 128.646us
           DATA_STREAM_SINK_OPERATOR (dest_id=1):
              - BlocksProduced: sum 57, avg 1, max 2, min 1
              - CloseTime: avg 567ns, max 1.301us, min 207ns
              - ExecTime: avg 44.522us, max 123.58us, min 17.818us
              - InitTime: avg 17.673us, max 90.800us, min 8.48us
              - InputRows: sum 366, avg 7, max 45, min 0
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 15.386us, max 55.426us, min 1.593us
              - OverallThroughput: sum 0.0 /sec, avg 0.0 /sec, max 0.0 /sec, min 0.0 /sec
              - WaitForDependencyTime: avg 0ns, max 0ns, min 0ns
                - WaitForRpcBufferQueue: avg 0ns, max 0ns, min 0ns
           LOCAL_EXCHANGE_OPERATOR (PASSTHROUGH) (id=-1):
              - BlocksProduced: sum 9, avg 0, max 1, min 0
              - CloseTime: avg 0ns, max 0ns, min 0ns
              - ExecTime: avg 3.967us, max 43.245us, min 875ns
              - GetBlockFailedTime: sum 8, avg 0, max 1, min 0
              - InitTime: avg 1.119us, max 5.384us, min 403ns
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 2.25 KB, avg 48.00 B, max 256.00 B, min 0.00 
              - OpenTime: avg 0ns, max 0ns, min 0ns
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 366, avg 7, max 45, min 0
              - WaitForDependency[LOCAL_EXCHANGE_OPERATOR_DEPENDENCY]Time: avg 194.951ms, max 195.317ms, min 194.726ms
         Pipeline : 1(instance_num=1):
            - WaitWorkerTime: avg 388.894us, max 388.894us, min 388.894us
           LOCAL_EXCHANGE_SINK_OPERATOR (PASSTHROUGH) (id=-1):
              - CloseTime: avg 97.855us, max 97.855us, min 97.855us
              - ExecTime: avg 166.265us, max 166.265us, min 166.265us
              - InitTime: avg 3.6us, max 3.6us, min 3.6us
              - InputRows: sum 366, avg 366, max 366, min 366
              - MemoryUsage: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - MemoryUsagePeak: sum 0.00 , avg 0.00 , max 0.00 , min 0.00 
              - OpenTime: avg 1.128us, max 1.128us, min 1.128us
              - WaitForDependency[LOCAL_EXCHANGE_SINK_DEPENDENCY]Time: avg 0ns, max 0ns, min 0ns
           OLAP_SCAN_OPERATOR (id=0. nereids_id=2963. table name = date_dim(date_dim)):
              - PlanInfo
                 - TABLE: tpcds.date_dim(date_dim), PREAGGREGATION: ON
                 - PREDICATES: (d_year = 2000)
                 - partitions=1/1 (date_dim)
                 - tablets=9/9, tabletList=10109,10111,10113 ...
                 - cardinality=73049, avgRowSize=123.370346, numNodes=1
                 - pushAggOp=NONE
                 - projections: d_date_sk
                 - project output tuple id: 1
              - BlocksProduced: sum 9, avg 9, max 9, min 9
              - CloseTime: avg 85.728us, max 85.728us, min 85.728us
              - ExecTime: avg 194.989ms, max 194.989ms, min 194.989ms
              - InitTime: avg 107.862us, max 107.862us, min 107.862us
              - MemoryUsage: sum 768.00 B, avg 768.00 B, max 768.00 B, min 768.00 B
              - MemoryUsagePeak: sum 1.50 KB, avg 1.50 KB, max 1.50 KB, min 1.50 KB
              - OpenTime: avg 147.560ms, max 147.560ms, min 147.560ms
              - ProjectionTime: avg 0ns, max 0ns, min 0ns
              - RowsProduced: sum 366, avg 366, max 366, min 366
              - RuntimeFilterInfo: sum , avg , max , min 
              - WaitForDependency[OLAP_SCAN_OPERATOR_DEPENDENCY]Time: avg 46.759ms, max 46.759ms, min 46.759ms
