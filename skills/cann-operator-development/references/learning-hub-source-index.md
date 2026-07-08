# GitCode CANN Learning Hub Source Index

Source read with Kimi WebBridge from:

- `https://gitcode.com/cann/cann-learning-hub/tree/master/tutorials/ascendc_operator_development`
- Main tree commit observed on 2026-07-07: `16b769a025591232d170216e00cf70d6047ed430`

Use this file only for provenance and notebook routing. For task work, prefer the chapter-specific reference files.

## Course Structure

The page contains 41 published notebooks:

1. `01_basic_overview/01.01_chapter_intro.ipynb`
2. `01_basic_overview/01.02_ai_and_operator_basics.ipynb`
3. `01_basic_overview/01.03_cann_arch_ascend_npu_principle.ipynb`
4. `01_basic_overview/01.04_ascend_c_op_dev_basic_concepts.ipynb`
5. `01_basic_overview/01.05_chapter_practice.ipynb`
6. `02_AscendC_basic/02.01_chapter_intro.ipynb`
7. `02_AscendC_basic/02.02_HelloWorld.ipynb`
8. `02_AscendC_basic/02.03_ascendc_programming_paradigm_and_api_introduction.ipynb`
9. `02_AscendC_basic/02.04_introduction_to_kernel_functions_based_on_add_operator.ipynb`
10. `02_AscendC_basic/02.05_chapter_test.ipynb`
11. `03_intermediate_vector_operator_development/03.01_chapter_intro.ipynb`
12. `03_intermediate_vector_operator_development/03.02_operator_engineering_intro.ipynb`
13. `03_intermediate_vector_operator_development/03.03_acl_pybind_call.ipynb`
14. `03_intermediate_vector_operator_development/03.04_generalized_tiling_design.ipynb`
15. `03_intermediate_vector_operator_development/03.05_tiling_template_attr_tbuf_workspace.ipynb`
16. `03_intermediate_vector_operator_development/03.06_chapter_practice.ipynb`
17. `04_matmul_basic/04.01_chapter_intro.ipynb`
18. `04_matmul_basic/04.02_matrix_multiplication_introduction.ipynb`
19. `04_matmul_basic/04.03_matrix_multiplication_operator_development_with_high_level_api.ipynb`
20. `04_matmul_basic/04.04_chapter_test.ipynb`
21. `05_fused_operator_development/05.01_chapter_intro.ipynb`
22. `05_fused_operator_development/05.02_fused_operator_concept_intro.ipynb`
23. `05_fused_operator_development/05.03_vv_fused_operator_development.ipynb`
24. `05_fused_operator_development/05.04_cv_fused_operator_development.ipynb`
25. `05_fused_operator_development/05.05_chapter_practice.ipynb`
26. `06_opensource_repo_operator_intro_and_contribution/06.01_chapter_intro.ipynb`
27. `06_opensource_repo_operator_intro_and_contribution/06.02_opensource_repo_intro_and_verification.ipynb`
28. `06_opensource_repo_operator_intro_and_contribution/06.03_operator_development_based_on_opensource_repo.ipynb`
29. `06_opensource_repo_operator_intro_and_contribution/06.04_ut_st_writing_and_verification.ipynb`
30. `06_opensource_repo_operator_intro_and_contribution/06.05_chapter_practice.ipynb`
31. `07_Troubleshooting/07.01_chapter_intro.ipynb`
32. `07_Troubleshooting/07.02_CPU_Debugging_Overview.ipynb`
33. `07_Troubleshooting/07.03_NPU_On-Board_Debugging.ipynb`
34. `07_Troubleshooting/07.04_Typical_Issues_in_AscendC_Operator_Development.ipynb`
35. `07_Troubleshooting/07.05_chapter_test.ipynb`
36. `08_performance_optimization/08.01_chapter_intro.ipynb`
37. `08_performance_optimization/08.02_profiling_tool_usage.ipynb`
38. `08_performance_optimization/08.03_simulation_analysis.ipynb`
39. `08_performance_optimization/08.04_ascendc_op_perf_optimization_demo.ipynb`
40. `08_performance_optimization/08.05_chapter_practice.ipynb`
41. `09_course_practice/09.01_vector_ops_practice.ipynb`

## Reference Mapping

- Chapters 1-2: `ascendc-learning-hub-basics.md`
- Chapter 3: `intermediate-vector-operator-development.md`, `generalized-tiling-strategy.md`
- Chapter 4: `matmul-operator-development.md`
- Chapter 5: `fused-operator-development.md`
- Chapter 6: `opensource-operator-repo.md`
- Chapter 7: `troubleshooting-debugging.md`
- Chapter 8: `performance-optimization.md`
- Chapter 9: `course-practice-logsigmoid.md`

## Retrieval Notes

The GitCode page exposes each notebook through `scanFilePath` links on the main directory page. The notebook blob pages render the raw `.ipynb` JSON in a code block, so Kimi WebBridge can read each file by navigating or fetching:

`https://gitcode.com/cann/cann-learning-hub/blob/master/<notebook-path>`

Do not load every notebook into context for ordinary tasks. Use the task routing in `SKILL.md` and the condensed chapter reference files first.
