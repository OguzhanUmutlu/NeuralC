#include <node_api.h>
#include <iostream>
#include "network.h"
#include "main.h"

char *read_string(napi_env _, napi_value value)
{
    size_t length;
    NC(_, napi_get_value_string_utf8(_, value, nullptr, 0, &length), "Couldn't read the length of a string.", nullptr);
    char *buffer = new char[length + 1];
    NC(_, napi_get_value_string_utf8(_, value, buffer, length + 1, nullptr), "Couldn't read the value of a string.", nullptr);
    return buffer;
}

int32_t read_int32(napi_env _, napi_value value)
{
    int32_t res;
    NC(_, napi_get_value_int32(_, value, &res), "Couldn't parse the value to int32_t.", 0);
    return res;
}

int64_t read_int64(napi_env _, napi_value value)
{
    int64_t res;
    NC(_, napi_get_value_int64(_, value, &res), "Couldn't parse the value to int64_t.", 0);
    return res;
}

double read_double(napi_env _, napi_value value)
{
    double res;
    NC(_, napi_get_value_double(_, value, &res), "Couldn't parse the value to double.", 0);
    return res;
}

bool read_bool(napi_env _, napi_value value)
{
    bool res;
    NC(_, napi_get_value_bool(_, value, &res), "Couldn't parse the value to bool.", 0);
    return res;
}

napi_value create_int32(napi_env _, int32_t value)
{
    napi_value res;
    NC(_, napi_create_int32(_, value, &res), "Couldn't create an int32 value.", nullptr);
    return res;
}

napi_value create_int64(napi_env _, int64_t value)
{
    napi_value res;
    NC(_, napi_create_int64(_, value, &res), "Couldn't create an int64 value.", nullptr);
    return res;
}
napi_value create_double(napi_env _, double value)
{
    napi_value res;
    NC(_, napi_create_double(_, value, &res), "Couldn't create an double value.", nullptr);
    return res;
}

napi_value create_bool(napi_env _, bool value)
{
    napi_value res;
    NC(_, napi_get_boolean(_, value, &res), "Couldn't create a bool value.", nullptr);
    return res;
}

napi_value create_object(napi_env _)
{
    napi_value res;
    NC(_, napi_create_object(_, &res), "Couldn't create an object value.", nullptr);
    return res;
}

napi_value create_array(napi_env _)
{
    napi_value res;
    NC(_, napi_create_array(_, &res), "Couldn't create an array value.", nullptr);
    return res;
}

napi_value create_array_with_length(napi_env _, size_t length)
{
    napi_value res;
    NC(_, napi_create_array_with_length(_, length, &res), "Couldn't create an array value.", nullptr);
    return res;
}

napi_value create_function(napi_env _, napi_callback fn)
{
    napi_value res;
    NC(_, napi_create_function(_, nullptr, 0, fn, nullptr, &res), "Couldn't create a function.", nullptr);
    return res;
}

napi_value create_undefined(napi_env _)
{
    napi_value res;
    NC(_, napi_get_undefined(_, &res), "Couldn't get the undefined itself.", nullptr);
    return res;
}

int set_object_key(napi_env _, napi_value obj, const char *key, napi_value value)
{
    NC(_, napi_set_named_property(_, obj, key, value), "Couldn't set the key of an object value.", 0);
    return 0;
}

napi_value get_object_key(napi_env _, napi_value obj, const char *key)
{
    napi_value res;
    NC(_, napi_get_named_property(_, obj, key, &res), "Couldn't get the key of an object.", nullptr);
    return res;
}

bool has_object_key(napi_env _, napi_value obj, const char *key)
{
    bool res;
    NC(_, napi_has_named_property(_, obj, key, &res), "Couldn't check the key of an object.", NULL);
    return res;
}

bool is_array(napi_env _, napi_value val)
{
    bool res;
    NC(_, napi_is_array(_, val, &res), "Couldn't check if a napi_value was array or not.", NULL);
    return res;
}

uint32_t get_array_length(napi_env _, napi_value arr)
{
    uint32_t res;
    NC(_, napi_get_array_length(_, arr, &res), "Couldn't find the length of an array.", NULL);
    return res;
}

int prepare_args(napi_env _, napi_callback_info info, size_t &argc, napi_value *args)
{
    NC(_, napi_get_cb_info(_, info, &argc, args, nullptr, nullptr), "Couldn't parse the arguments.", 0);
    return 0;
}

int prepare_this(napi_env _, napi_callback_info info, napi_value &_this)
{
    NC(_, napi_get_cb_info(_, info, nullptr, nullptr, &_this, nullptr), "Couldn't parse the arguments.", 0);
    return 0;
}

int prepare_args_this(napi_env _, napi_callback_info info, size_t &argc, napi_value *args, napi_value &_this)
{
    NC(_, napi_get_cb_info(_, info, &argc, args, &_this, nullptr), "Couldn't parse the arguments.", 0);
    return 0;
}

void add_function_to_object(napi_env _, napi_value object, const char *name, napi_callback cb)
{
    set_object_key(_, object, name, create_function(_, cb));
}

void inspect(size_t size, unsigned char *bytePtr)
{
    for (size_t i = 0; i < size; i++)
    {
        for (int j = 7; j >= 0; j--)
        {
            std::cout << ((bytePtr[i] >> j) & 1);
        }
        std::cout << ' ';
    }
    std::cout << "\n";
}

napi_value call_function(napi_env _, napi_value func, size_t argc, const napi_value *args)
{
    napi_value glob;
    NC(_, napi_get_global(_, &glob), "Couldn't get the global context.", nullptr);
    napi_value res;
    NC(_, napi_call_function(_, glob, func, argc, args, &res), "Function call failed.", nullptr);
    return res;
}

napi_value init(napi_env _, napi_value exports)
{
    napi_value object = create_object(_);
    add_function_to_object(_, object, "NeuralNetwork", neuralNetworkConstructor);
    return object;
}

NAPI_MODULE(_, init);