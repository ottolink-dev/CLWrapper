include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB TESTS LIST_DIRECTORIES true "*")
foreach(item ${TESTS})
	if(IS_DIRECTORY ${item})
		add_subdirectory(${item})
	endif()
endforeach()
