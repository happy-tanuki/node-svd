// C part for SVD
extern "C" {
#include "svdlib.h"
}

#include "node.h"
#include "v8.h"


namespace node_svd {

using v8::Exception;
using v8::FunctionCallbackInfo;
using v8::Isolate;
using v8::Context;
using v8::Local;
using v8::Number;
using v8::Object;
using v8::String;
using v8::Value;
using v8::Array;
using v8::NewStringType;

/**
 * Svd computation using svdlibc
 * 
 * @param args [A, dim, {U, V, debug}]
 * @return {d, U, S, V}
 */
void Svd(const FunctionCallbackInfo<Value>& args) {

    Isolate* isolate = args.GetIsolate();
    Local<Context> context = isolate->GetCurrentContext();

    // parsing the arguments
    int rows = 0, cols = 0;
    int dim = 0; // number of dimensions, 0=all by default
    bool useU = true, useV = false; // untranspose state
    bool merr = false;
    Local<Array> m = Array::New(isolate, 0);
    switch (args.Length()) {
        case 4: // with debug level
            if (!args[3]->IsNumber()) {
                isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Debug type must be a number!", NewStringType::kNormal).ToLocalChecked()));
                return;
            }
            
        case 3: // with settings
            if (!args[2]->IsObject()) {
                isolate->ThrowException(Exception::Error(String::NewFromUtf8(isolate, "Settings must be an object!", NewStringType::kNormal).ToLocalChecked()));
                return;
            }else{
                Local<Object> settings = args[2]->ToObject(context).ToLocalChecked();
                // debug flag
                if(settings->Has(context, String::NewFromUtf8(isolate, "debug", NewStringType::kNormal).ToLocalChecked()).ToChecked()){
                    Local<Value> debugValue = settings->Get(context, String::NewFromUtf8(isolate, "debug", NewStringType::kNormal).ToLocalChecked()).ToLocalChecked();
                    if(debugValue->IsNumber()){
                        SVDVerbosity = long(debugValue->NumberValue(context).ToChecked());
                    }
                }
                // useU
                if(settings->Has(context, String::NewFromUtf8(isolate, "U", NewStringType::kNormal).ToLocalChecked()).ToChecked()){
                    Local<Value> uValue = settings->Get(context, String::NewFromUtf8(isolate, "U", NewStringType::kNormal).ToLocalChecked()).ToLocalChecked();
                    if(uValue->IsBoolean()){
                        useU = uValue->BooleanValue(isolate);
                    }
                }
                // useV
                if(settings->Has(context, String::NewFromUtf8(isolate, "V", NewStringType::kNormal).ToLocalChecked()).ToChecked()){
                    Local<Value> vValue = settings->Get(context, String::NewFromUtf8(isolate, "V", NewStringType::kNormal).ToLocalChecked()).ToLocalChecked();
                    if(vValue->IsBoolean()){
                        useV = vValue->BooleanValue(isolate);
                    }
                }
            }
        case 2: // with dimension
            if (!args[1]->IsNumber()) {
                isolate->ThrowException(Exception::Error(String::NewFromUtf8(isolate, "Dimension must be a number!", NewStringType::kNormal).ToLocalChecked()));
                return;
            }
            dim = int(args[1]->NumberValue(context).ToChecked());
        case 1: // only A
            if (!args[0]->IsArray()) {
                merr = true;
            } else {
                m = Local<Array>::Cast(args[0]);
                rows = m->Length();
                if (rows == 0) {
                    merr = true;
                } else {
                    Local<Value> v0 = m->Get(context, Number::New(isolate, 0)).ToLocalChecked();
                    if (!v0->IsArray()) merr = true;
                    else {
                        Local<Array> r0 = Local<Array>::Cast(v0);
                        cols = r0->Length();
                        if (cols == 0) merr = true;
                    }
                }
            }
            if (merr) {
                isolate->ThrowException(Exception::Error(String::NewFromUtf8(isolate, "First argument must be a non-empty matrix of number!", NewStringType::kNormal).ToLocalChecked()));
                return;
            }
            break;
        default:
            isolate->ThrowException(Exception::Error(String::NewFromUtf8(isolate, "Requires at least a matrix argument!", NewStringType::kNormal).ToLocalChecked()));
            return;
    }

    // 1 = create matrix and populate it
    DMat D = svdNewDMat(rows, cols);
    for (int y = 0; y < rows; ++y) {
        Local<Array> row = Local<Array>::Cast(m->Get(context, y).ToLocalChecked());
        for (int x = 0; x < cols; ++x) {
            D->value[y][x] = row->Get(context, x).ToLocalChecked()->NumberValue(context).ToChecked();
        }
    }
    // sparse matrix generation
    SMat A = svdConvertDtoS(D);
    svdFreeDMat(D); // we can release the dense matrix

    // 2 = svd !
    SVDRec rec = svdLAS2A(A, dim);
    svdFreeSMat(A);

    // 3 = wrap result
    Local<Object> res = Object::New(isolate);
    // Dimension
    res->Set(context, String::NewFromUtf8(isolate, "d", NewStringType::kNormal).ToLocalChecked(), Number::New(isolate, dim = rec->d));
    // Ut vector
    Local<Array> U;
    if (useU) {
        // let's untranspose
        U = Array::New(isolate, rows);
        for (int y = 0; y < rows; ++y) {
            Local<Array> row = Array::New(isolate, dim);
            for (int x = 0; x < dim; ++x) {
                row->Set(context, x, Number::New(isolate, rec->Ut->value[x][y]));
            }
            U->Set(context, y, row);
        }
    } else {
        U = Array::New(isolate, dim);
        for (int x = 0; x < dim; ++x) {
            Local<Array> trow = Array::New(isolate, rows);
            for (int y = 0; y < rows; ++y) {
                trow->Set(context, y, Number::New(isolate, rec->Ut->value[x][y]));
            }
            U->Set(context, x, trow);
        }
    }
    res->Set(context, String::NewFromUtf8(isolate, "U", NewStringType::kNormal).ToLocalChecked(), U);
    // Singular values
    Local<Array> S = Array::New(isolate, rec->d);
    for (int s = 0; s < rec->d; ++s) S->Set(context, s, Number::New(isolate, rec->S[s]));
    res->Set(context, String::NewFromUtf8(isolate, "S", NewStringType::kNormal).ToLocalChecked(), S);
    // Vt vector
    Local<Array> V;
    if (useV) {
        // let's untranspose
        V = Array::New(isolate, cols);
        for (int y = 0; y < cols; ++y) {
            Local<Array> row = Array::New(isolate, dim);
            for (int x = 0; x < dim; ++x) {
                row->Set(context, x, Number::New(isolate, rec->Vt->value[x][y]));
            }
            V->Set(context, y, row);
        }
    } else {
        V = Array::New(isolate, dim);
        for (int x = 0; x < dim; ++x) {
            Local<Array> trow = Array::New(isolate, cols);
            for (int y = 0; y < cols; ++y) {
                trow->Set(context, y, Number::New(isolate, rec->Vt->value[x][y]));
            }
            V->Set(context, x, trow);
        }
    }
    res->Set(context, String::NewFromUtf8(isolate, "V", NewStringType::kNormal).ToLocalChecked(), V);
    // release result memory finally
    svdFreeSVDRec(rec);

    args.GetReturnValue().Set(res);
}

void InitSVD(Local<Object> exports) {
    NODE_SET_METHOD(exports, "svd", Svd);
}

NODE_MODULE(svd, InitSVD)

}