#pragma once

#include <string>
#include <vector>

namespace colmap {
	enum class CopyType { COPY, HARD_LINK, SOFT_LINK };

	// Append trailing slash to string if it does not yet end with a slash.
	std::string EnsureTrailingSlash(const std::string& str);

	// Check whether file name has the file extension (case insensitive).
	bool HasFileExtension(const std::string& file_name,
		const std::string& ext);

	// Split the path into its root and extension, for example,
	// "dir/file.jpg" into "dir/file" and ".jpg".
	void SplitFileExtension(const std::string& path,
		std::string* root,
		std::string* ext);

	// Copy or link file from source to destination path
	void FileCopy(const std::string& src_path,
		const std::string& dst_path,
		CopyType type = CopyType::COPY);

	// Check if the path points to an existing directory.
	bool ExistsFile(const std::string& path);

	// Check if the path points to an existing directory.
	bool ExistsDir(const std::string& path);

	// Check if the path points to an existing file or directory.
	bool ExistsPath(const std::string& path);

	// Create the directory if it does not exist.
	void CreateDirIfNotExists(const std::string& path,
		bool recursive = false);

	// Extract the base name of a path, e.g., "image.jpg" for "/dir/image.jpg".
	std::string GetPathBaseName(const std::string& path);

	// Get the path of the parent directory for the given path.
	std::string GetParentDir(const std::string& path);

	// Join dir and file_name into one path.
	std::string JoinPath(const std::string& dir,
		const std::string& file_name);

	// Join dir, base_name and ext into one path.
	std::string JoinPath(const std::string& dir,
		const std::string& base_name,
		const std::string& ext);

	// Return list of files in directory.
	std::vector<std::string> GetFileList(const std::string& path);

	// Return list of files, recursively in all sub-directories.
	std::vector<std::string> GetRecursiveFileList(
		const std::string& path);

	// Return list of directories, recursively in all sub-directories.
	std::vector<std::string> GetDirList(const std::string& path);

	// Return list of directories, recursively in all sub-directories.
	std::vector<std::string> GetRecursiveDirList(
		const std::string& path);

	// Get the size in bytes of a file.
	int64_t GetFileSize(const std::string& path);

}