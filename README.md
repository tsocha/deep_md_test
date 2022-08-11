# deep_md_test
Deepmd integration with Paddle, without lammps

- Used Paddle commit `c91aaced74aa1a34c8bde2e53b3072baf8012e73` and compile with MKLDNN ON to generate paddle_inference_install_dir and 
```
export PADDLE_ROOT=/xxx/Paddle/build/paddle_inference_install_dir
```
- Compile deepmd_root by 
```
rm -rf /home/danqing/deepmdroot/ && mkdir /home/danqing/deepmdroot && DEEPMD_ROOT=/home/danqing/deepmdroot(or add in bashrc with export)
cd /home/danqing/repo/paddle-deepmd/source && rm -rf build && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$DEEPMD_ROOT -DPADDLE_ROOT=$PADDLE_ROOT -DUSE_CUDA_TOOLKIT=FALSE -DFLOAT_PREC=low ..
make -j 4 && make install
make lammps
```
- Compile deep_md_test and run
```
bash run.sh
./infer_test
```
- The results are as follows
```
I0809 16:18:27.501428 382150 naive_executor.cc:110] ---  skip [feed], feed -> type
I0809 16:18:27.501431 382150 naive_executor.cc:110] ---  skip [feed], feed -> coord
I0809 16:18:27.503818 382150 naive_executor.cc:110] ---  skip [save_infer_model/scale_0.tmp_1], fetch -> fetch
I0809 16:18:27.503829 382150 naive_executor.cc:110] ---  skip [save_infer_model/scale_1.tmp_1], fetch -> fetch
I0809 16:18:27.503830 382150 naive_executor.cc:110] ---  skip [save_infer_model/scale_2.tmp_1], fetch -> fetch
I0809 16:18:27.503832 382150 naive_executor.cc:110] ---  skip [save_infer_model/scale_3.tmp_1], fetch -> fetch
I0809 16:18:27.503837 382150 naive_executor.cc:110] ---  skip [save_infer_model/scale_4.tmp_1], fetch -> fetch
I0809 16:18:27.503840 382150 naive_executor.cc:110] ---  skip [save_infer_model/scale_5.tmp_1], fetch -> fetch
I0809 16:18:27.503844 382150 naive_executor.cc:110] ---  skip [save_infer_model/scale_6.tmp_1], fetch -> fetch
...
I0809 16:18:27.849522 382150 infer_test.cc:139] output[0]: -93.6274
I0809 16:18:27.849589 382150 infer_test.cc:139] output[100]: -187.135

```
