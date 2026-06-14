## Modified DSA Versions

This repository includes two modified versions of the Depth-Scanning Algorithm (DSA), adapted for real-data applications in different study regions.

### 1. `DSA_WCSB_HY.py`

`DSA_WCSB_HY.py` is the version adapted for real-data focal-depth determination in the Western Canadian Sedimentary Basin (WCSB).

**Modified by:**  
Hongyu Yu (`hongyu.yu@zju.edu.cn`)  
**Modification date:** May 2022

**Modification notes:**  
This version was adapted for focal-depth determination of earthquakes in the Western Canadian Sedimentary Basin (WCSB). The main modifications include adjustments to the direct-arrival phase extraction strategy and the preliminary depth-range selection strategy. These changes were made to improve the robustness of the Depth-Scanning Algorithm (DSA) for real-data applications in the WCSB.

---

### 2. `DSA_SX_parallel.py`

`DSA_SX_parallel.py` is a parallelized version developed based on `DSA_WCSB_HY.py`, and was further adapted for earthquake focal-depth determination in the Shanxi Graben area.

**Modified by:**  
Hongyu Yu, Shaoqi Zhang(3230100342@qq.com), and Yanjiu Wu(40433155@qq.com)  
**Modification date:** March 2026

**Modification notes:**  
Based on the WCSB-adapted version, this version further improves the computational efficiency of the DSA workflow. The main modification is the parallelization of the station and depth loops in Step 3, preliminary focal-depth determination, and Step 4, final solution based on travel-time residuals. These changes reduce runtime while preserving the original depth-scanning logic and output format.

---

### Summary of Modifications

| File | Study region | Main purpose | Main modifications |
|---|---|---|---|
| `DSA_WCSB_HY.py` | Western Canadian Sedimentary Basin | Real-data focal-depth determination | Adjusted direct-arrival phase extraction and preliminary depth-range selection |
| `DSA_SX_parallel.py` | Shanxi Graben | Parallelized real-data focal-depth determination | Parallelized station and depth loops in Steps 3 and 4 |
