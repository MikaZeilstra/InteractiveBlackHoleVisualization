#pragma once
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include <fstream>
#include <iomanip>

template <class T>
class Archive {
public:
	static void serialize(std::string filename, T& instance) {
		{
			std::ofstream ofs(filename, std::ios::out | std::ios::binary);
			cereal::BinaryOutputArchive archive(ofs);
			archive(instance);
		}
	};

	static bool load(std::string filename, T& instance) {
		// Try loading existing grid file, if fail compute new grid.
		std::ifstream ifs(filename, std::ios::in | std::ios::binary);

		if (ifs.good()) {
			{
				// Create an input archive
				cereal::BinaryInputArchive iarch(ifs);
				iarch(instance);
			}
			return true;
		}
		return false;
	}

};