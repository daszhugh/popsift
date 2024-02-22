#include "path.h"

#include "string.h"

#include <fstream>

// clang-format off
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
// clang-format on

namespace colmap {

	std::string EnsureTrailingSlash(const std::string& str) {
		if (str.length() > 0) {
			if (str.back() != '/') {
				return str + "/";
			}
		}
		else {
			return str + "/";
		}
		return str;
	}

	bool HasFileExtension(const std::string& file_name, const std::string& ext) {
		std::string ext_lower = ext;
		StringToLower(&ext_lower);
		if (file_name.size() >= ext_lower.size() &&
			file_name.substr(file_name.size() - ext_lower.size(), ext_lower.size()) ==
			ext_lower) {
			return true;
		}
		return false;
	}

	void SplitFileExtension(const std::string& path,
		std::string* root,
		std::string* ext) {
		const auto parts = StringSplit(path, ".");
		if (parts.size() == 1) {
			*root = parts[0];
			*ext = "";
		}
		else {
			*root = "";
			for (size_t i = 0; i < parts.size() - 1; ++i) {
				*root += parts[i] + ".";
			}
			*root = root->substr(0, root->length() - 1);
			if (parts.back() == "") {
				*ext = "";
			}
			else {
				*ext = "." + parts.back();
			}
		}
	}

	void FileCopy(const std::string& src_path,
		const std::string& dst_path,
		CopyType type) {
		switch (type) {
		case CopyType::COPY:
			boost::filesystem::copy_file(src_path, dst_path);
			break;
		case CopyType::HARD_LINK:
			boost::filesystem::create_hard_link(src_path, dst_path);
			break;
		case CopyType::SOFT_LINK:
			boost::filesystem::create_symlink(src_path, dst_path);
			break;
		}
	}

	bool ExistsFile(const std::string& path) {
		return boost::filesystem::is_regular_file(path);
	}

	bool ExistsDir(const std::string& path) {
		return boost::filesystem::is_directory(path);
	}

	bool ExistsPath(const std::string& path) {
		return boost::filesystem::exists(path);
	}

	void CreateDirIfNotExists(const std::string& path, bool recursive) {
		if (ExistsDir(path)) {
			return;
		}
		if (recursive) {
			boost::filesystem::create_directories(path);
		}
		else {
			boost::filesystem::create_directory(path);
		}
	}

	std::string GetPathBaseName(const std::string& path) {
		const std::vector<std::string> names =
			StringSplit(StringReplace(path, "\\", "/"), "/");
		if (names.size() > 1 && names.back() == "") {
			return names[names.size() - 2];
		}
		else {
			return names.back();
		}
	}

	std::string GetParentDir(const std::string& path) {
		return boost::filesystem::path(path).parent_path().string();
	}

	std::string JoinPath(const std::string& dir, const std::string& file_name) {
		return (boost::filesystem::path(dir) / boost::filesystem::path(file_name))
			.string();
	}

	std::string JoinPath(const std::string& dir,
		const std::string& base_name,
		const std::string& ext) {
		if (ext.empty()) {
			return JoinPath(dir, base_name);
		}

		return JoinPath(dir, base_name + ext);
	}

	std::vector<std::string> GetFileList(const std::string& path) {
		std::vector<std::string> file_list;
		for (auto it = boost::filesystem::directory_iterator(path);
			it != boost::filesystem::directory_iterator();
			++it) {
			if (boost::filesystem::is_regular_file(*it)) {
				const boost::filesystem::path file_path = *it;
				file_list.push_back(file_path.string());
			}
		}
		return file_list;
	}

	std::vector<std::string> GetRecursiveFileList(const std::string& path) {
		std::vector<std::string> file_list;
		for (auto it = boost::filesystem::recursive_directory_iterator(path);
			it != boost::filesystem::recursive_directory_iterator();
			++it) {
			if (boost::filesystem::is_regular_file(*it)) {
				const boost::filesystem::path file_path = *it;
				file_list.push_back(file_path.string());
			}
		}
		return file_list;
	}

	std::vector<std::string> GetDirList(const std::string& path) {
		std::vector<std::string> dir_list;
		for (auto it = boost::filesystem::directory_iterator(path);
			it != boost::filesystem::directory_iterator();
			++it) {
			if (boost::filesystem::is_directory(*it)) {
				const boost::filesystem::path dir_path = *it;
				dir_list.push_back(dir_path.string());
			}
		}
		return dir_list;
	}

	std::vector<std::string> GetRecursiveDirList(const std::string& path) {
		std::vector<std::string> dir_list;
		for (auto it = boost::filesystem::recursive_directory_iterator(path);
			it != boost::filesystem::recursive_directory_iterator();
			++it) {
			if (boost::filesystem::is_directory(*it)) {
				const boost::filesystem::path dir_path = *it;
				dir_list.push_back(dir_path.string());
			}
		}
		return dir_list;
	}

	int64_t GetFileSize(const std::string& path) {
		std::ifstream file(path, std::ifstream::ate | std::ifstream::binary);
		return file.tellg();
	}
}
