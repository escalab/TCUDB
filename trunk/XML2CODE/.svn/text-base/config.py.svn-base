#!/usr/bin/python
"""
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
"""

"""
Several configurable variables:
    @CODETYPE determines whether CUDA or OpenCL codes should be generated.
    0 represents CUDA and 1 represents OpenCL.

    @joinType determines whether we should generate invisible joins for star
    schema queries. 0 represents normal join and 1 represents invisible join.

    @POS describes where the data are stored in the host memory and how the
    codes should be generated. 0 means data are stored in pageable host
    memory and data are explicitly transferred. 1 means data are stored in
    pinned host memory and data are explicitly transferred. 2 means data are
    stored in pinned host memory and the kernel will directly access the data
    without explicit data transferring. 3 means data are stored in disk and only
    mapped to host memory.

    @SOA is currently for testing only.
"""

joinType = 0    
POS = 0
CODETYPE = 0
SOA = 0

"""
OpenCL specific configurable variables: 
    @PID is the platform ID that will execute the query.
    @DTYPE specifies the type of the device which executes the query. 
    0 represents CL_DEVICE_TYPE_GPU,
    1 represnets CL_DEVICE_TYPE_CPU and
    2 represnets CL_DEVICE_TYPE_ACCELERATOR.
"""

PID = 0
DTYPE = 0

