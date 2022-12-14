#pragma once
#include <vector>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

/**
* Created by Thomas on 8/9/14.
* This class is used to create perfect spatial hashing for a set of 3D indices.
* This class takes as input a list of 3d index (3D integer vector), and creates a mapping that can be used to pair a 3d index with a value.
* The best part about this type of hash map is that it can compress 3d spatial data in such a way that there spatial coherency (3d indices near each other have paired values near each other in the hash table) and the lookup time is O(1).
* Since it's perfect hashing there is no hash collisions
* This hashmap could be used for a very efficient hash table on the GPU due to coherency and only 2 lookups from a texture-hash-table would be needed, one for the offset to help create the hash, and one for the actual value indexed by the final hash.
* This implementation is based off the paper: http://hhoppe.com/perfecthash.pdf, Perfect Spatial Hashing by Sylvain Lefebvre &Hugues Hopp, Microsoft Research
*
*  To use:
*  accumulate your spatial data in a list to pass to the PSHOffsetTable class
*  construct the table with the list
*  you now use this class just as your "mapping", it has the hash function for your hash table
*  create your 3D hash with the chosen width from PSHOffsetTable.hashTableWidth.
*  Then to get the index into your hash table, just use PSHOffsetTable.hash(key).
*  That's it.
*
*  If you want to update the offsetable, you can do so by using the updateOffsets() with the modified list of spatial data.
*/

class OffsetBucket {
public:
	bool create = false;
	std::vector<cv::Point2i> contents;
	std::vector<cv::Point3f> data;
	cv::Point2i index; //index in offset table
	OffsetBucket() {};

	OffsetBucket(cv::Point2i index) : index(index) {
		create = true;
	}
};

class PSHOffsetTable {
private:

	friend class cereal::access;
	template < class Archive >
	void serialize(Archive & ar) {
		ar(hashTable, offsetTable, hashPosTag, hashTableWidth, offsetTableWidth);
	}

	std::vector<cv::Point2i> elements;
	std::vector<cv::Point3f> datapoints;
	std::vector<OffsetBucket> offsetBuckets;
	std::vector<bool> hashFilled;

	int offsetFindLimit = 120;
	int tableCreateLimit = 10;
	int creationAttempts = 0;

	int calcHashTableWidth(int size) {
		float d = (float)pow(size*1.1f, 1.f / 2.f);
		return (int)(d + 1.1f);
	}

	int gcd(int a, int b) {
		if (b == 0)
			return a;
		return gcd(b, a % b);
	}

	int calcOffsetTableWidth(int size) {
		float d = (float)pow(size / 4.f, 1.f / 2.f);
		int width = (int)(d + 1.1f);

		while (gcd(width, hashTableWidth) > 1) { //make sure there are no common factors
			width++;
		}
		return width;
	}

	void tryCreateAgain() {
		creationAttempts++;
		if (creationAttempts >= tableCreateLimit) {
			std::cout << "WRONG" << std::endl;
		}
		resizeOffsetTable();
		clearFilled();
		calculateOffsets();
	}

	bool checkForBadCollisions(OffsetBucket bucket) {
		std::vector<cv::Point2i> testList;
		for (int i = 0; i < bucket.contents.size(); i++) {
			cv::Point2i ele = bucket.contents[i];
			cv::Point2i hash = hash0(ele);
			for (int q = 0; q < testList.size(); q++) {
				if (testList[q].x == hash.x && testList[q].y == hash.y) {
					return true;
				}
			}
			testList.push_back(hash);
		}
		return false;
	}

	void fillHashCheck(OffsetBucket bucket, cv::Point2i offset) {
		for (int i = 0; i < bucket.contents.size(); i++) {
			cv::Point2i ele = bucket.contents[i];
			cv::Point2i _hash = hashFunc(ele, offset);
			if (hashFilled[_hash.x * hashTableWidth + _hash.y]) std::cout << "ALREADY FILLED" << std::endl;
			hashFilled[_hash.x * hashTableWidth + _hash.y] = true;
			hashTable[(_hash.x * hashTableWidth + _hash.y) * 3] = bucket.data[i].x;
			hashTable[(_hash.x * hashTableWidth + _hash.y) * 3 + 1] = bucket.data[i].y;
			hashTable[(_hash.x * hashTableWidth + _hash.y) * 3 + 2] = bucket.data[i].z;

			hashPosTag[(_hash.x * hashTableWidth + _hash.y) * 2] = ele.x;
			hashPosTag[(_hash.x * hashTableWidth + _hash.y) * 2 + 1] = ele.y;

			// Fill hashtable itself as well over here
		}
	}

	bool findBadOffset(cv::Point2i index, std::vector<cv::Point2i>& badOffsets, OffsetBucket bucket, cv::Point2i& offset) {
		index = hash1(index);
		offset = { offsetTable[(index.x * offsetTableWidth + index.y) * 2], 
				   offsetTable[(index.x * offsetTableWidth + index.y) * 2 + 1] };
		for (int q = 0; q < badOffsets.size(); q++) {
			if (badOffsets[q].x == offset.x && badOffsets[q].y == offset.y)
				return false;
		}
		if (OffsetWorks(bucket, offset)) {
			return true;
		}
		badOffsets.push_back(offset);
	}

	bool findOffsetRandom(OffsetBucket bucket, cv::Point2i& newOffset) {
		//vector<cv::Point2i> badOffsets;
		//cv::Point2i offset;
		//cv::Point2i index = { bucket.index.x + 1, bucket.index.y };
		//if (findBadOffset(index, badOffsets, bucket, offset)) {
		//	newOffset = offset;
		//	return true;
		//}
		//index = { bucket.index.x, bucket.index.y + 1 };
		//if (findBadOffset(index, badOffsets, bucket, offset)) {
		//	newOffset = offset;
		//	return true;
		//}
		//index = { bucket.index.x -1, bucket.index.y };
		//if (findBadOffset(index, badOffsets, bucket, offset)) {
		//	newOffset = offset;
		//	return true;
		//}
		//index = { bucket.index.x, bucket.index.y - 1 };
		//if (findBadOffset(index, badOffsets, bucket, offset)) {
		//	newOffset = offset;
		//	return true;
		//}
		if (bucket.contents.size() == 1) {
			cv::Point2i hashIndex;
			if (findAEmptyHash(hashIndex)) {
				newOffset = { hashIndex.x - hash0(bucket.contents[0]).x, hashIndex.y - hash0(bucket.contents[0]).y};
				return true;
			}
			else return false;
		}

		cv::Point2i seed = { (rand() % hashTableWidth) - hashTableWidth / 2,
					  (rand() % hashTableWidth) - hashTableWidth / 2 };
		for (int i = 0; i <= 5; i++) {
			for (int x = i; x < hashTableWidth; x += 5) {
				for (int y = i; y < hashTableWidth; y += 5) {
					cv::Point2i offset = { seed.x + x, seed.y + y };
					if (OffsetWorks(bucket, offset)) {
						newOffset = offset;
						return true;
					}
					//cv::Point2i index = hash0(index);
					//if (!hashFilled[index.x * hashTableWidth + index.y]) {
					//	cv::Point2i offset = { index.x - hash0(bucket.contents[0]).x, index.y - hash0(bucket.contents[0]).y };
				}
			}
		}

		return false;
	}

	bool findAEmptyHash(cv::Point2i &index) {
		cv::Point2i seed = { (rand() % hashTableWidth) - hashTableWidth / 2, (rand() % hashTableWidth) - hashTableWidth / 2 };
		for (int x = 0; x < hashTableWidth; x++) {
			for (int y = 0; y < hashTableWidth; y++) {
				index = { seed.x + x, seed.y + y };
				index = hash0(index);
				if (!hashFilled[index.x * hashTableWidth + index.y]) return true;
			}
		}
		return false;
	}

	cv::Point2i findAEmptyHash2(cv::Point2i start) {
		for (int x = 0; x < hashTableWidth; x++) {
			for (int y = 0; y < hashTableWidth; y++) {
				if (x + y == 0) continue;
				cv::Point2i index = { start.x + x, start.y + y };
				index = hash0(index);
				if (!hashFilled[index.x * hashTableWidth + index.y]) return index;
			}
		}
	}

	bool OffsetWorks(OffsetBucket bucket, cv::Point2i offset) {
		for (int i = 0; i < bucket.contents.size(); i++) {
			cv::Point2i ele = bucket.contents[i];
			cv::Point2i _hash = hashFunc(ele, offset);
			if (hashFilled[_hash.x * hashTableWidth + _hash.y]) {
				return false;
			}
		}
		return true;
	}

	cv::Point2i hash1(cv::Point2i key) {
		return{ (key.x + offsetTableWidth) % offsetTableWidth, (key.y + offsetTableWidth) % offsetTableWidth };
	}

	cv::Point2i hash0(cv::Point2i key) {
		return{ (key.x + hashTableWidth) % hashTableWidth, (key.y + hashTableWidth) % hashTableWidth };
	}



	cv::Point2i hashFunc(cv::Point2i key, cv::Point2i offset) {
		cv::Point2i add = { hash0(key).x + offset.x, hash0(key).y + offset.y };
		return hash0(add);
	}

	void resizeOffsetTable() {
		offsetTableWidth += 5; //test
		while (gcd(offsetTableWidth, hashTableWidth % offsetTableWidth) > 1) {
			offsetTableWidth++;
		}
		offsetBuckets = std::vector<OffsetBucket>(offsetTableWidth*offsetTableWidth);
		offsetTable = std::vector<int>(offsetTableWidth*offsetTableWidth*2);
		clearOffsstsToZero();
	}

	void clearFilled() {
		for (int x = 0; x < hashTableWidth; x++) {
			for (int y = 0; y < hashTableWidth; y++) {
				hashFilled[x*hashTableWidth + y] = false;
			}
		}
	}

	void clearOffsstsToZero() {
		for (int x = 0; x < offsetTableWidth; x++) {
			for (int y = 0; y < offsetTableWidth; y++) {
				offsetTable[(x*offsetTableWidth + y) * 2] = 0;
				offsetTable[(x*offsetTableWidth + y) * 2 + 1] = 0;
			}
		}
	}

	void cleanUp() {
		//elements = 0;
		//offsetBuckets = 0;
		//hashFilled = 0;
	}

	void putElementsIntoBuckets() {
		for (int i = 0; i < n; i++) {
			cv::Point2i ele = elements[i];
			cv::Point2i index = hash1(ele);
			if (!offsetBuckets[index.x * offsetTableWidth + index.y].create) {
				offsetBuckets[index.x * offsetTableWidth + index.y] = OffsetBucket(index);
			}
			offsetBuckets[index.x * offsetTableWidth + index.y].contents.push_back(ele);
			offsetBuckets[index.x * offsetTableWidth + index.y].data.push_back(datapoints[i]);
		}
	}

	std::vector<OffsetBucket> createSortedBucketList() {

		std::vector<OffsetBucket> bucketList;

		for (int x = 0; x < offsetTableWidth; x++) { //put the buckets into the bucketlist and sort
			for (int y = 0; y < offsetTableWidth; y++) {
				if (offsetBuckets[x*offsetTableWidth+y].create) {
					bucketList.push_back(offsetBuckets[x*offsetTableWidth + y]);
				}
			}
		}
		quicksort(bucketList, 0, bucketList.size() - 1);
		return bucketList;
	}

	void calculateOffsets() {

		putElementsIntoBuckets();
		std::vector<OffsetBucket> bucketList = createSortedBucketList();

		for (int i = 0; i < bucketList.size(); i++) {
			OffsetBucket bucket = bucketList[i];
			cv::Point2i offset;
			//if (checkForBadCollisions(bucket)) {
			//	cout << "badcollisions" << endl;
			//}

			bool succes = findOffsetRandom(bucket, offset);

			if (!succes) {
				tryCreateAgain();
				break;
			}
			offsetTable[(bucket.index.x * offsetTableWidth + bucket.index.y) * 2] = offset.x;
			offsetTable[(bucket.index.x * offsetTableWidth + bucket.index.y) * 2 + 1] = offset.y;

			fillHashCheck(bucket, offset);

		}

	}




public:
	std::vector<int> offsetTable; // used to be [][]
	std::vector<float> hashTable;
	std::vector<int> hashPosTag;

	int offsetTableWidth;
	int hashTableWidth;
	int n;

	cv::Point2i hashFunc(cv::Point2i key) {
		cv::Point2i index = hash1(key);
		cv::Point2i add = { hash0(key).x + offsetTable[(index.x*offsetTableWidth + index.y)*2],
					 hash0(key).y + offsetTable[(index.x*offsetTableWidth + index.y)*2+1] };
		return hash0(add);
	}

	//Random random = new Random(System.currentTimeMillis());

	PSHOffsetTable() {};

	

	PSHOffsetTable(std::vector<cv::Point2i>& _elements, std::vector<cv::Point3f>& _datapoints) {
		srand(time(NULL));
   		//srand(1);

		int size = _elements.size();
		n = size;
		hashTableWidth = calcHashTableWidth(size);
		offsetTableWidth = calcOffsetTableWidth(size);

		hashFilled = std::vector<bool>(hashTableWidth*hashTableWidth);
		hashTable = std::vector<float>(hashTableWidth*hashTableWidth*3);
		hashPosTag = std::vector<int>(hashTableWidth*hashTableWidth*2);
		
		offsetBuckets = std::vector<OffsetBucket>(offsetTableWidth*offsetTableWidth);

		offsetTable = std::vector<int>(offsetTableWidth*offsetTableWidth*2);
		clearOffsstsToZero();
		elements = _elements;
		datapoints = _datapoints;

		calculateOffsets();


		//crappy solution for now:
		//offsetTable1 = vector<int>(offsetTableWidth*offsetTableWidth*2);
		//hashTable1 = vector<float>(hashTableWidth*hashTableWidth * 2);
		//hashPosTag1 = vector<int>(hashTableWidth*hashTableWidth * 2);

		//for (int i = 0; i < offsetTable.size(); i++) {
		//	offsetTable1[2 * i] = offsetTable[i].x;
		//	offsetTable1[2 * i + 1] = offsetTable[i].y;
		//}

		//for (int i = 0; i < hashTable.size(); i++) {
		//	hashTable1[2 * i] = hashTable[i].x;
		//	hashTable1[2 * i + 1] = hashTable[i].y;
		//	hashPosTag1[2 * i] = hashPosTag[i].x;
		//	hashPosTag1[2 * i + 1] = hashPosTag[i].y;
		//}

		//check
		//for (int i = 0; i < n; i++) {
		//	cv::Point2i index = hashFunc(elements[i]);
		//	cv::Point2f hashtab = hashTable[index.x*hashTableWidth + index.y];
		//	if (hashtab.x != datapoints[i].x || hashtab.y != datapoints[i].y) cout << "mistakes were made" << endl;
		//	cv::Point2i postag = hashPosTag[index.x*hashTableWidth + index.y];
		//	if (postag.x != elements[i].x || postag.y != elements[i].y) cout << "other mistakes were made" << endl;
		//}
		//cleanUp();
	};

	void quicksort(std::vector<OffsetBucket>& bucketList, int start, int end) {
		int i = start;
		int j = end;
		int pivot = bucketList[start + (end - start) / 2].contents.size();
		while (i <= j) {
			while (bucketList[i].contents.size() > pivot) {
				i++;
			}
			while (bucketList[j].contents.size() < pivot) {
				j--;
			}
			if (i <= j) {
				OffsetBucket temp = bucketList[i];
				bucketList[i] = bucketList[j];
				bucketList[j] = temp;
				i++;
				j--;
			}

		}
		if (start < j)
			quicksort(bucketList, start, j);
		if (i < end)
			quicksort(bucketList, i, end);
	}

};

//void updateOffsets(vector<cv::Point2i> _elements, vector<cv::Point2f> _dataPoints) {
//	int size = _elements.size();
//	n = size;
//	hashTableWidth = calcHashTableWidth(size);
//	int oldOffsetWidth = offsetTableWidth;
//	offsetTableWidth = calcOffsetTableWidth(size); //this breaks if original creation didn't use initial table calculated width
//
//	hashFilled = vector<bool>(hashTableWidth*hashTableWidth);
//	hashTable = vector<float>(hashTableWidth*hashTableWidth);
//	hashPosTag = vector<int>(hashTableWidth*hashTableWidth);
//	offsetBuckets = vector<OffsetBucket>(offsetTableWidth * offsetTableWidth);
//
//	elements = _elements;
//	datapoints = _dataPoints;
//
//	if (oldOffsetWidth != offsetTableWidth) {
//		offsetTable = vector<int>(offsetTableWidth*offsetTableWidth);
//		clearOffsstsToZero();
//		calculateOffsets();
//		cleanUp();
//	}
//	else {
//		putElementsIntoBuckets();
//		vector<OffsetBucket> bucketList = createSortedBucketList();
//
//		for (int i = 0; i < bucketList.size(); i++) {
//			OffsetBucket bucket = bucketList[i];
//			cv::Point2i offset = offsetTable[bucket.index.x*offsetTableWidth + bucket.index.y];
//			if (!OffsetWorks(bucket, offset)) {
//				bool succes = findOffsetRandom(bucket, offset);
//				if (!succes) {
//					cout << "AGAIN" << endl;
//					tryCreateAgain();
//					break;
//				}
//				offsetTable[bucket.index.x*offsetTableWidth + bucket.index.y] = offset;
//			}
//			fillHashCheck(bucket, offset);
//			//            if(checkForBadCollisions(bucket)) {
//			//                tryCreateAgain();
//			//                break;
//			//            }
//		}
//	}
//
//
//}