#ifdef ENABLE_COREML

#include <vector>
#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import "VirtualOrch/CoreMLWrapper.h"
#import "VirtualOrchModel.h"

const void* loadModel(const char* modelPath) {
    NSString* modelPathStr = [[NSString alloc] initWithUTF8String:modelPath];
    NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];

    NSError *error;
    const void* model = CFBridgingRetain([[VirtualOrchModel alloc] initWithContentsOfURL:modelURL error:&error]);
    return model;
}

void predictWith(const void* model, const std::vector<float>& input_ids, std::vector<float>& logits) {
    NSInteger seqLength = input_ids.size();
    MLMultiArray *inMultiArray = [[MLMultiArray alloc] initWithDataPointer: (void*)input_ids.data()
                                                                      shape: @[@(seqLength)]
                                                                   dataType: MLMultiArrayDataTypeFloat32
                                                                   strides: @[@1]
                                                                deallocator: nil
                                                                      error: nil];

    VirtualOrchModelOutput *modelOutput = [(__bridge id)model predictionFromInput_ids:inMultiArray error:nil];
    MLMultiArray *outMA = modelOutput.linear_0;

    cblas_scopy(55028,
                (float*)outMA.dataPointer + 55028 * (seqLength - 1), 1,
                logits.data(), 1);
}

void closeModel(const void* model) {
    if (model) {
        CFRelease(model);
    }
}

#endif