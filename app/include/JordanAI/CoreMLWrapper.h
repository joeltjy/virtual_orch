#pragma once
#ifdef ENABLE_COREML

#include <vector>

const void *loadModel(const char *modelPath);

void closeModel(const void *model);

void predictWith(const void *model, const std::vector<float> &input_ids, std::vector<float> &logits);
#endif
