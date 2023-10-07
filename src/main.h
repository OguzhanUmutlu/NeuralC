#include <node_api.h>

#define NC(env, call, err, ret)                                              \
    do                                                                       \
    {                                                                        \
        napi_status status = (call);                                         \
        if (status != napi_ok)                                               \
        {                                                                    \
            const napi_extended_error_info *error_info = NULL;               \
            napi_get_last_error_info(env, &error_info);                      \
            bool is_pending;                                                 \
            napi_is_exception_pending(env, &is_pending);                     \
            if (!is_pending)                                                 \
            {                                                                \
                const char *message = (error_info->error_message == NULL)    \
                                          ? ("C++: " #call " Message: " err) \
                                          : error_info->error_message;       \
                NAPI_ERROR(env, message);                                    \
                return ret;                                                  \
            }                                                                \
        }                                                                    \
    } while (0)

#define NAPI_ERROR(env, message)                                         \
    std::cout << "A C++ Exception has been thrown: " << message << "\n"; \
    napi_throw_error(env, NULL, message);

#define DO_UNWRAP(obj, any, ret) NC(env, napi_unwrap(env, obj, reinterpret_cast<void **>(&any)), "Couldn't unwrap the data from the function call. Probably because the function was used as a variable and called afterward.", ret)

#define DO_WRAP(obj, store, type, ret) NC(                                                                                     \
    env, napi_wrap(                                                                                                            \
             env, obj, store, [](napi_env env, void *data, void *hint) { delete static_cast<type>(data); }, nullptr, nullptr), \
    "Couldn't wrap the neural network.", ret)

#define EXPECT_TYPE(any, type, ret)                                        \
    do                                                                     \
    {                                                                      \
        napi_valuetype T;                                                  \
        napi_typeof(env, any, &T);                                         \
        if (T != type)                                                     \
        {                                                                  \
            NAPI_ERROR(env, "Expected '" #any "' to be a type of " #type); \
            return ret;                                                    \
        }                                                                  \
    } while (0)

char *read_string(napi_env env, napi_value value);
int32_t read_int32(napi_env env, napi_value value);
int64_t read_int64(napi_env env, napi_value value);
double read_double(napi_env env, napi_value value);
bool read_bool(napi_env env, napi_value value);
napi_status get_type_of(napi_env env, napi_value value);
napi_value create_int32(napi_env env, int32_t value);
napi_value create_int64(napi_env env, int64_t value);
napi_value create_double(napi_env env, double value);
napi_value create_bool(napi_env env, bool value);
napi_value create_object(napi_env env);
napi_value create_array(napi_env env);
napi_value create_array_with_length(napi_env env, size_t length);
napi_value create_function(napi_env env, napi_callback fn);
napi_value create_undefined(napi_env env);
napi_value get_object_key(napi_env env, napi_value obj, const char *key);
int set_object_key(napi_env env, napi_value obj, const char *name, napi_value value);
bool has_object_key(napi_env env, napi_value obj, const char *key);
bool is_array(napi_env env, napi_value val);
uint32_t get_array_length(napi_env env, napi_value arr);
int prepare_args(napi_env env, napi_callback_info info, size_t &argc, napi_value *args);
int prepare_this(napi_env env, napi_callback_info info, napi_value &_this);
int prepare_args_this(napi_env env, napi_callback_info info, size_t &argc, napi_value *args, napi_value &_this);
void add_function_to_object(napi_env env, napi_value object, const char *name, napi_callback cb);
void inspect(size_t size, unsigned char *bytePtr);
napi_value call_function(napi_env env, napi_value func, size_t argc, const napi_value *args);
napi_value init(napi_env env, napi_value exports);