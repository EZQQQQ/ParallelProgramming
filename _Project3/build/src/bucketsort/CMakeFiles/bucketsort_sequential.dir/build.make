# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /nfsmnt/123100001/CSC4005-2023Fall/project3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /nfsmnt/123100001/CSC4005-2023Fall/project3/build

# Include any dependencies generated for this target.
include src/bucketsort/CMakeFiles/bucketsort_sequential.dir/depend.make

# Include the progress variables for this target.
include src/bucketsort/CMakeFiles/bucketsort_sequential.dir/progress.make

# Include the compile flags for this target's objects.
include src/bucketsort/CMakeFiles/bucketsort_sequential.dir/flags.make

src/bucketsort/CMakeFiles/bucketsort_sequential.dir/sequential.cpp.o: src/bucketsort/CMakeFiles/bucketsort_sequential.dir/flags.make
src/bucketsort/CMakeFiles/bucketsort_sequential.dir/sequential.cpp.o: ../src/bucketsort/sequential.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfsmnt/123100001/CSC4005-2023Fall/project3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/bucketsort/CMakeFiles/bucketsort_sequential.dir/sequential.cpp.o"
	cd /nfsmnt/123100001/CSC4005-2023Fall/project3/build/src/bucketsort && /opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bucketsort_sequential.dir/sequential.cpp.o -c /nfsmnt/123100001/CSC4005-2023Fall/project3/src/bucketsort/sequential.cpp

src/bucketsort/CMakeFiles/bucketsort_sequential.dir/sequential.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bucketsort_sequential.dir/sequential.cpp.i"
	cd /nfsmnt/123100001/CSC4005-2023Fall/project3/build/src/bucketsort && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfsmnt/123100001/CSC4005-2023Fall/project3/src/bucketsort/sequential.cpp > CMakeFiles/bucketsort_sequential.dir/sequential.cpp.i

src/bucketsort/CMakeFiles/bucketsort_sequential.dir/sequential.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bucketsort_sequential.dir/sequential.cpp.s"
	cd /nfsmnt/123100001/CSC4005-2023Fall/project3/build/src/bucketsort && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfsmnt/123100001/CSC4005-2023Fall/project3/src/bucketsort/sequential.cpp -o CMakeFiles/bucketsort_sequential.dir/sequential.cpp.s

src/bucketsort/CMakeFiles/bucketsort_sequential.dir/__/utils.cpp.o: src/bucketsort/CMakeFiles/bucketsort_sequential.dir/flags.make
src/bucketsort/CMakeFiles/bucketsort_sequential.dir/__/utils.cpp.o: ../src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfsmnt/123100001/CSC4005-2023Fall/project3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/bucketsort/CMakeFiles/bucketsort_sequential.dir/__/utils.cpp.o"
	cd /nfsmnt/123100001/CSC4005-2023Fall/project3/build/src/bucketsort && /opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bucketsort_sequential.dir/__/utils.cpp.o -c /nfsmnt/123100001/CSC4005-2023Fall/project3/src/utils.cpp

src/bucketsort/CMakeFiles/bucketsort_sequential.dir/__/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bucketsort_sequential.dir/__/utils.cpp.i"
	cd /nfsmnt/123100001/CSC4005-2023Fall/project3/build/src/bucketsort && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfsmnt/123100001/CSC4005-2023Fall/project3/src/utils.cpp > CMakeFiles/bucketsort_sequential.dir/__/utils.cpp.i

src/bucketsort/CMakeFiles/bucketsort_sequential.dir/__/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bucketsort_sequential.dir/__/utils.cpp.s"
	cd /nfsmnt/123100001/CSC4005-2023Fall/project3/build/src/bucketsort && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfsmnt/123100001/CSC4005-2023Fall/project3/src/utils.cpp -o CMakeFiles/bucketsort_sequential.dir/__/utils.cpp.s

# Object files for target bucketsort_sequential
bucketsort_sequential_OBJECTS = \
"CMakeFiles/bucketsort_sequential.dir/sequential.cpp.o" \
"CMakeFiles/bucketsort_sequential.dir/__/utils.cpp.o"

# External object files for target bucketsort_sequential
bucketsort_sequential_EXTERNAL_OBJECTS =

src/bucketsort/bucketsort_sequential: src/bucketsort/CMakeFiles/bucketsort_sequential.dir/sequential.cpp.o
src/bucketsort/bucketsort_sequential: src/bucketsort/CMakeFiles/bucketsort_sequential.dir/__/utils.cpp.o
src/bucketsort/bucketsort_sequential: src/bucketsort/CMakeFiles/bucketsort_sequential.dir/build.make
src/bucketsort/bucketsort_sequential: src/bucketsort/CMakeFiles/bucketsort_sequential.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/nfsmnt/123100001/CSC4005-2023Fall/project3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable bucketsort_sequential"
	cd /nfsmnt/123100001/CSC4005-2023Fall/project3/build/src/bucketsort && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bucketsort_sequential.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/bucketsort/CMakeFiles/bucketsort_sequential.dir/build: src/bucketsort/bucketsort_sequential

.PHONY : src/bucketsort/CMakeFiles/bucketsort_sequential.dir/build

src/bucketsort/CMakeFiles/bucketsort_sequential.dir/clean:
	cd /nfsmnt/123100001/CSC4005-2023Fall/project3/build/src/bucketsort && $(CMAKE_COMMAND) -P CMakeFiles/bucketsort_sequential.dir/cmake_clean.cmake
.PHONY : src/bucketsort/CMakeFiles/bucketsort_sequential.dir/clean

src/bucketsort/CMakeFiles/bucketsort_sequential.dir/depend:
	cd /nfsmnt/123100001/CSC4005-2023Fall/project3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /nfsmnt/123100001/CSC4005-2023Fall/project3 /nfsmnt/123100001/CSC4005-2023Fall/project3/src/bucketsort /nfsmnt/123100001/CSC4005-2023Fall/project3/build /nfsmnt/123100001/CSC4005-2023Fall/project3/build/src/bucketsort /nfsmnt/123100001/CSC4005-2023Fall/project3/build/src/bucketsort/CMakeFiles/bucketsort_sequential.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/bucketsort/CMakeFiles/bucketsort_sequential.dir/depend

