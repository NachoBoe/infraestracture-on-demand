# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/panda/Projects/dynamic_experiments

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/panda/Projects/dynamic_experiments/build

# Include any dependencies generated for this target.
include CMakeFiles/mosek_solution.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mosek_solution.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mosek_solution.dir/flags.make

CMakeFiles/mosek_solution.dir/experiment.cpp.o: CMakeFiles/mosek_solution.dir/flags.make
CMakeFiles/mosek_solution.dir/experiment.cpp.o: ../experiment.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/panda/Projects/dynamic_experiments/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mosek_solution.dir/experiment.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mosek_solution.dir/experiment.cpp.o -c /home/panda/Projects/dynamic_experiments/experiment.cpp

CMakeFiles/mosek_solution.dir/experiment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mosek_solution.dir/experiment.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/panda/Projects/dynamic_experiments/experiment.cpp > CMakeFiles/mosek_solution.dir/experiment.cpp.i

CMakeFiles/mosek_solution.dir/experiment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mosek_solution.dir/experiment.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/panda/Projects/dynamic_experiments/experiment.cpp -o CMakeFiles/mosek_solution.dir/experiment.cpp.s

# Object files for target mosek_solution
mosek_solution_OBJECTS = \
"CMakeFiles/mosek_solution.dir/experiment.cpp.o"

# External object files for target mosek_solution
mosek_solution_EXTERNAL_OBJECTS =

mosek_solution: CMakeFiles/mosek_solution.dir/experiment.cpp.o
mosek_solution: CMakeFiles/mosek_solution.dir/build.make
mosek_solution: libFusionCXX.a
mosek_solution: /home/panda/MOSEK/mosek/10.0/tools/platform/linux64x86/bin/libmosek64.so
mosek_solution: CMakeFiles/mosek_solution.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/panda/Projects/dynamic_experiments/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mosek_solution"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mosek_solution.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mosek_solution.dir/build: mosek_solution

.PHONY : CMakeFiles/mosek_solution.dir/build

CMakeFiles/mosek_solution.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mosek_solution.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mosek_solution.dir/clean

CMakeFiles/mosek_solution.dir/depend:
	cd /home/panda/Projects/dynamic_experiments/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/panda/Projects/dynamic_experiments /home/panda/Projects/dynamic_experiments /home/panda/Projects/dynamic_experiments/build /home/panda/Projects/dynamic_experiments/build /home/panda/Projects/dynamic_experiments/build/CMakeFiles/mosek_solution.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mosek_solution.dir/depend

