# Collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build each source file independently
foreach(source ${srcs})
  get_filename_component(name ${source} NAME_WE)

  # caffe target already exits
  if(name MATCHES "caffe")
    set(name ${name}.bin)
  endif()

  # target
  add_executable(${name} ${source})
  target_link_libraries(${name} ${Caffe_LINK})
  caffe_default_properties(${name})

  # set back RUNTIME_OUTPUT_DIRECTORY
  caffe_set_runtime_directory(${name} "${PROJECT_BINARY_DIR}/tools")
  caffe_set_solution_folder(${name} tools)

  # restore output name without suffix
  if(name MATCHES "caffe.bin")
    set_target_properties(${name} PROPERTIES OUTPUT_NAME caffe)
  endif()

  # Install
  install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})

endforeach(source)

## ---------------------- contrib tools ---------------------------
# Collect source files
file(GLOB_RECURSE srcs ${PROJECT_SOURCE_DIR}/contrib/*/caffe/tools/*.cpp)

# Build each source file independently
foreach(source ${srcs})
  get_filename_component(name ${source} NAME_WE)
  get_filename_component(path ${source} PATH)

  # target
  add_executable(${name} ${source})
  target_link_libraries(${name} ${Caffe_LINK})
  caffe_default_properties(${name})

  # set back RUNTIME_OUTPUT_DIRECTORY
  caffe_set_runtime_directory(${name} "${PROJECT_BINARY_DIR}/tools/contrib")
  caffe_set_solution_folder(${name} tools/contrib)

  if(UNIX OR APPLE)
    string(REPLACE "caffe/tools" "tutorial/tools" contrib_tool_dir ${path})
    set(__linkname ${contrib_tool_dir}/${name})
    add_custom_command(TARGET ${name} POST_BUILD
                       COMMAND ln -sf $<TARGET_FILE:${name}> "${__linkname}"
                       COMMENT "Creating symlink ${__linkname} -> ${PROJECT_BINARY_DIR}/tools/contrib/${name}")
  endif()

  # Install
  install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR}/contrib)

endforeach(source)