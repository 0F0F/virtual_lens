cmake_minimum_required(VERSION 2.8)
project( virtual_lens )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package( OpenCV REQUIRED )
add_executable ( virtual_lens test.h test.cpp virtual_lens.cpp )
target_link_libraries( virtual_lens ${OpenCV_LIBS} )
