#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
struct Record {
    std::string                         id;    // unique identifier
    std::unordered_set<std::string>     tags;  // set for fast lookup
    long long views{0}, likes{0}, comments{0}; // engagement stats
};

// ---------------------------------------------------------------------------
// Split helper  —  returns vector of non‑empty substrings
static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream       ss(s);
    std::string             buf;
    while (std::getline(ss, buf, delim))
        if (!buf.empty()) out.push_back(buf);
    return out;
}

// Jaccard distance between two tag sets  (0 ⇒ identical, 1 ⇒ no overlap)
static float jaccard(const std::unordered_set<std::string>& A,
                     const std::unordered_set<std::string>& B) {
    if (A.empty() && B.empty()) return 0.f;
    size_t inter = 0;
    for (const auto& t : A) if (B.count(t)) ++inter;
    size_t uni = A.size() + B.size() - inter;
    return 1.f - static_cast<float>(inter) / static_cast<float>(uni);
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <K> <dataset.csv> \"tag1;tag2;...\"\n";
        return 1;
    }
    // -------------------------------------------------------------------
    int K = std::atoi(argv[1]);
    if (K <= 0) {
        std::cerr << "[ERROR]  K must be positive.\n";
        return 1;
    }

    // -------------------------------------------------------------------
    // 1) Load dataset
    std::ifstream fin(argv[2]);
    if (!fin) {
        std::cerr << "[ERROR]  Cannot open dataset: " << argv[2] << "\n";
        return 1;
    }

    std::vector<Record> videos;
    std::string         line;

    // Skip header row if the first char is not a digit
    std::getline(fin, line);
    if (!line.empty() && !std::isdigit(static_cast<unsigned char>(line[0]))) {
        // header skipped
    } else {
        // rewind — the first line is already a data row
        fin.seekg(0);
    }

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto cols = split(line, ',');
        if (cols.size() < 5) continue;    // malformed row

        Record r;
        r.id = cols[0];
        for (auto& t : split(cols[1], ';')) r.tags.insert(t);
        r.views    = std::stoll(cols[2]);
        r.likes    = std::stoll(cols[3]);
        r.comments = std::stoll(cols[4]);
        videos.push_back(std::move(r));
    }
    if (videos.empty()) {
        std::cerr << "[ERROR]  Dataset is empty or malformed.\n";
        return 1;
    }

    // -------------------------------------------------------------------
    // 2) Parse query tags
    std::unordered_set<std::string> qTags;
    for (auto& t : split(argv[3], ';')) qTags.insert(t);
    if (qTags.empty()) {
        std::cerr << "[ERROR]  Provide at least one tag for the query.\n";
        return 1;
    }

    // -------------------------------------------------------------------
    // 3) Compute Jaccard distances to every video
    struct Node { float dist; size_t idx; };
    std::vector<Node> nbrs; nbrs.reserve(videos.size());

    for (size_t i = 0; i < videos.size(); ++i) {
        float d = jaccard(qTags, videos[i].tags);
        nbrs.push_back({d, i});
    }

    if (K > static_cast<int>(nbrs.size())) K = static_cast<int>(nbrs.size());

    // Partially sort so first K elements are the smallest by distance
    std::nth_element(nbrs.begin(), nbrs.begin() + K, nbrs.end(),
                     [](const Node& a, const Node& b) { return a.dist < b.dist; });

    std::sort(nbrs.begin(), nbrs.begin() + K,
              [](const Node& a, const Node& b) { return a.dist < b.dist; });

    // -------------------------------------------------------------------
    // 4) Aggregate stats over K neighbours
    long long sumViews = 0, sumLikes = 0, sumComments = 0;
    for (int i = 0; i < K; ++i) {
        const Record& v = videos[nbrs[i].idx];
        sumViews    += v.views;
        sumLikes    += v.likes;
        sumComments += v.comments;
    }

    long long pViews    = sumViews    / K;
    long long pLikes    = sumLikes    / K;
    long long pComments = sumComments / K;

    // -------------------------------------------------------------------
    // 5) Output
    std::cout << "Predicted → Views:"    << pViews
              << " Likes:"    << pLikes
              << " Comments:" << pComments << "\n";
    std::cout << "Top " << K << " neighbours:" << std::endl;
    for (int i = 0; i < K; ++i) {
        const Record& v = videos[nbrs[i].idx];
        std::cout << "  " << i + 1 << ". id=" << v.id
                  << " dist="       << nbrs[i].dist
                  << " " << v.views
                  << " " << v.likes
                  << " " << v.comments << std::endl;
    }

    return 0;
}