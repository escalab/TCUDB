/*
    Copyright (c) 2012-2013 The Ohio State University.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __GPU_OPENCLLIB__
#define __GPU_OPENCLLIB__
    #include <CL/cl.h>
    struct clContext{
        cl_context context;
        cl_command_queue queue;
        cl_program program;
        cl_kernel kernel;
    };
#endif
