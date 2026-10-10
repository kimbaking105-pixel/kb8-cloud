// ============================================================================
// CHESS ENGINE — build1.1-bitboard stages 1-3 (0x88 + incremental bitboard mirror)
// Original build1.1 control remains in engine_build11.cpp.
// CHESS ENGINE — MODULE 1: POSITION (0x88) + MODULE 2: MOVE GENERATOR
//               + MODULE 3: MAKE / UNMAKE + MODULE 4: CHECK / LEGAL MOVES
//               + MODULE 5: ZOBRIST HASH + FROZEN SEARCH A
//               + FROZEN EVALUATION A
//               + build0.2: P0 UCI lifecycle and draw correctness
//               + build0.3: canonical en-passant Zobrist hashing (Level A)
//               + build0.4: telemetry-only instrumentation (no search changes)
//               + build0.5: passed-pawn geometry correction only
//               + build0.6: exact UCI "go nodes N" infrastructure
//               + build0.8: SEE (Static Exchange Evaluation) — intermediate
//                 variant B: iterative swap algorithm WITH x-ray attackers,
//                 WITHOUT pin awareness (Level A philosophy, see build0.3).
//                 Integration: (1) capture ordering — SEE<0 captures are
//                 demoted below killers; (2) qsearch pruning — SEE<0
//                 captures are skipped when not in check (promotions and
//                 check evasions are never pruned). Re-implementation of the
//                 lost build0.8; verified by the new "seetest" suite.
//               + build0.9: futility pruning (negamax, depth<=2) + delta
//                 pruning (qsearch). Futility: at frontier/pre-frontier
//                 nodes in a null window, when static eval + margin
//                 {200,450} cannot reach alpha, quiet non-checking moves
//                 are skipped (never: in check / ply 0 / mate windows /
//                 first legal move / captures / promotions). Delta: in
//                 qsearch a capture is skipped when stand_pat + victim
//                 value + 200 cannot reach alpha (never: promotions /
//                 in check; disabled when total non-pawn material <=
//                 1300 to keep late-endgame resource-finding intact).
//                 Margins are coupled to Eval A noise (mine iter-1:
//                 MAE 128cp, P90 263cp) — re-check margins if eval changes.
//               + build1.0: time management (RELEASE). Soft/hard limits
//                 replace the flat "remaining/30 + inc" ration. Soft =
//                 remaining/25 + 0.7*inc — when exceeded, no NEW iteration
//                 is started. Hard = min(4*soft, remaining/4, remaining -
//                 50ms lag buffer) — enforced by the node-loop clock as
//                 before. Bestmove-stability factor scales the soft limit:
//                 stable for 1/2/3/4+ iterations -> 100/85/70/55%, move
//                 changed -> 125%, score dropped >40cp (fail-low) -> 135%
//                 (never past hard). Applies ONLY to wtime/btime clock
//                 searches: "go movetime/depth/nodes" and bench/tests are
//                 bit-identical to build0.9. movestogo intentionally not
//                 parsed (out of scope, documented).
// Single translation unit, C++17, no external libraries.
// ============================================================================
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
// ------------------------------- POSITION -----------------------------------
enum Piece : int { EMPTY = 0, PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6 };
enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };
constexpr int WK = 1, WQ = 2, BK = 4, BQ = 8;
constexpr int NO_SQ = 127;
constexpr int file_of(int sq) { return sq & 7; }
constexpr int rank_of(int sq) { return sq >> 4; }
constexpr int sq88(int f, int r) { return r * 16 + f; }
constexpr bool on_board(int sq) { return (sq & 0x88) == 0; }
constexpr bool valid_file(int f) { return (unsigned)f < 8; }
constexpr bool valid_rank(int r) { return (unsigned)r < 8; }
inline int piece_of(int pc) { return pc & 7; }
inline int color_of(int pc) { return pc >> 3; }
inline int make_pc(int p, int c) { return p | (c << 3); }

// -----------------------------------------------------------------------------
// build1.1-bitboard stages 1-3
//
// A 64-bit square is mapped as rank * 8 + file. Bitboards are authoritative;
// the compact 64-square piece lookup is maintained only for piece-at-square
// access and for the preserved 0x88 reference algorithms.
// -----------------------------------------------------------------------------
using Bitboard = uint64_t;

// Portable 64-bit bit operations. GCC/Clang expose __builtin_* names;
// MSVC uses the intrinsics below instead. All callers guarantee value != 0
// for the trailing-zero operation.
static inline int bb_ctz64(Bitboard value) {
#if defined(_MSC_VER)
    unsigned long index = 0;
#if defined(_WIN64)
    _BitScanForward64(&index, static_cast<unsigned __int64>(value));
    return static_cast<int>(index);
#else
    if (_BitScanForward(&index, static_cast<unsigned long>(value)))
        return static_cast<int>(index);
    _BitScanForward(&index, static_cast<unsigned long>(value >> 32));
    return static_cast<int>(index + 32);
#endif
#else
    return __builtin_ctzll(value);
#endif
}

static inline int bb_popcount64(Bitboard value) {
#if defined(_MSC_VER)
#if defined(_WIN64)
    return static_cast<int>(__popcnt64(static_cast<unsigned __int64>(value)));
#else
    return __popcnt(static_cast<unsigned int>(value)) +
        __popcnt(static_cast<unsigned int>(value >> 32));
#endif
#else
    return __builtin_popcountll(value);
#endif
}

constexpr int bb_sq_from_sq88(int sq) {
    return rank_of(sq) * 8 + file_of(sq);
}
constexpr int sq88_from_bb_sq(int sq) {
    return (sq >> 3) * 16 + (sq & 7);
}
constexpr Bitboard bb_square(int sq64) {
    return Bitboard(1) << sq64;
}

struct Position {
    // Bitboards are the authoritative occupancy/state representation. This
    // compact 64-square lookup is only a piece-at-square accelerator; there is
    // no 0x88 board storage in the active position.
    uint8_t piece[64];
    Bitboard bbPieces[COLOR_NB][7];
    Bitboard bbOccupied[COLOR_NB];
    Bitboard bbOccupiedAll;
    int sideToMove;
    int castle;
    int ep;
    int halfmove;
    int fullmove;
    int ksq[COLOR_NB]; // build1.1 candidate: cached king squares, derived from board
    uint64_t key;
};

static uint8_t bb_invalid_piece = EMPTY;
static inline uint8_t& bb_piece_ref(Position& p, int sq88) {
    return on_board(sq88) ? p.piece[bb_sq_from_sq88(sq88)] : bb_invalid_piece;
}
static inline const uint8_t& bb_piece_ref(const Position& p, int sq88) {
    return on_board(sq88) ? p.piece[bb_sq_from_sq88(sq88)] : bb_invalid_piece;
}
int findKing(const Position& p, int color);
static void clear_pos(Position& p) {
    std::memset(p.piece, 0, sizeof p.piece);
    p.sideToMove = WHITE;
    p.castle = 0;
    p.ep = NO_SQ;
    p.halfmove = 0;
    p.fullmove = 1;
    p.ksq[WHITE] = p.ksq[BLACK] = NO_SQ;
    std::memset(p.bbPieces, 0, sizeof p.bbPieces);
    std::memset(p.bbOccupied, 0, sizeof p.bbOccupied);
    p.bbOccupiedAll = 0;
    p.key = 0;
}

static void bb_clear(Position& p) {
    std::memset(p.bbPieces, 0, sizeof p.bbPieces);
    std::memset(p.bbOccupied, 0, sizeof p.bbOccupied);
    p.bbOccupiedAll = 0;
}

static void bb_add_piece(Position& p, int pc, int sq88) {
    if (!pc || !on_board(sq88)) return;
    const int c = color_of(pc), pt = piece_of(pc);
    const Bitboard b = bb_square(bb_sq_from_sq88(sq88));
    p.bbPieces[c][pt] |= b;
    p.bbOccupied[c] |= b;
    p.bbOccupiedAll |= b;
}

static void bb_remove_piece(Position& p, int pc, int sq88) {
    if (!pc || !on_board(sq88)) return;
    const int c = color_of(pc), pt = piece_of(pc);
    const Bitboard b = bb_square(bb_sq_from_sq88(sq88));
    p.bbPieces[c][pt] &= ~b;
    p.bbOccupied[c] &= ~b;
    p.bbOccupiedAll &= ~b;
}

static void bb_move_piece(Position& p, int pc, int from, int to) {
    bb_remove_piece(p, pc, from);
    bb_add_piece(p, pc, to);
}

static void bb_rebuild(Position& p) {
    bb_clear(p);
    for (int sq = 0; sq < 128; ++sq) {
        if (on_board(sq) && bb_piece_ref(p, sq))
            bb_add_piece(p, bb_piece_ref(p, sq), sq);
    }
}

static bool bb_piece_lookup_matches(const Position& p);

static bool bb_state_matches(const Position& p) {
    Position ref = p;
    bb_rebuild(ref);
    return std::memcmp(p.bbPieces, ref.bbPieces, sizeof p.bbPieces) == 0 &&
        std::memcmp(p.bbOccupied, ref.bbOccupied, sizeof p.bbOccupied) == 0 &&
        p.bbOccupiedAll == ref.bbOccupiedAll &&
        bb_piece_lookup_matches(p);
}

static bool bb_piece_lookup_matches(const Position& p) {
    for (int sq64 = 0; sq64 < 64; ++sq64) {
        const Bitboard bit = bb_square(sq64);
        int expected = EMPTY;
        for (int c = WHITE; c <= BLACK; ++c)
            for (int pt = PAWN; pt <= KING; ++pt)
                if (p.bbPieces[c][pt] & bit)
                    expected = make_pc(pt, c);
        if (p.piece[sq64] != expected) return false;
    }
    return true;
}

static uint64_t zPiece[16][128], zCastle[16], zEp[128], zSide;
static uint64_t splitmix64(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static void initZobrist() {
    static bool done = false;
    if (done) return;
    done = true;
    uint64_t s = 0x123456789abcdef0ULL;
    for (int pc = 0; pc < 16; ++pc)
        for (int sq = 0; sq < 128; ++sq)
            zPiece[pc][sq] = splitmix64(s);
    zSide = splitmix64(s);
    for (int i = 0; i < 16; ++i) zCastle[i] = splitmix64(s);
    for (int i = 0; i < 128; ++i) zEp[i] = splitmix64(s);
}
// =============================================================================
// build0.3: canonical en-passant Zobrist normalization — Level A.
//
// Previously computeHash()/makeMove()/the manual null-move code all folded the
// RAW p.ep square into the hash unconditionally. That is wrong in two ways:
//   1. Two positions that differ only by a "dead" ep square (no pawn is
//      actually able to capture en passant) hashed to different keys even
//      though they are the same chess position for all practical purposes
//      (TT probes, repetition, etc.). This inflates TT/repetition misses and
//      can even make a benign FEN look "different" from the canonical one
//      reached by playing the double push over the board.
//   2. A malformed/hand-written FEN could set an ep square with no capturing
//      pawn geometry at all (e.g., no enemy pawn to actually capture), and the
//      raw square still leaked into the hash.
//
// hash_ep_square() below is the single source of truth for "does the ep
// square actually matter for hashing purposes". It performs a cheap, purely
// local board check (no move generation, no recursion) and is used from
// computeHash(), makeMove()'s incremental update, and the manual null-move
// hash patch in search_negamax(), so all three stay consistent with each
// other by construction.
//
// This is intentionally "Level A": it only checks that a pawn of the side to
// move is geometrically adjacent to the capture square and that the captured
// pawn actually exists. It does NOT verify that the capture would be legal
// (e.g. it may still be pinned). Full legality (Level B) is out of scope for
// this pass — gen_pawn()/generateLegalMoves() already reject illegal en
// passant captures during move generation, this function only controls
// Zobrist hashing.
// =============================================================================
static int hash_ep_square(const Position& p) {
    if (p.ep == NO_SQ) return NO_SQ;
    const int us = p.sideToMove;
    const int r = rank_of(p.ep);
    // The ep square must sit on the rank a pawn lands on after a double push:
    // rank 5 (0-indexed) for White to move, rank 2 for Black to move.
    if (us == WHITE) {
        if (r != 5) return NO_SQ;
    }
    else {
        if (r != 2) return NO_SQ;
    }
    const int dir = (us == WHITE) ? 16 : -16;
    const int captured = p.ep - dir;
    if (!on_board(captured)) return NO_SQ;
    if (bb_piece_ref(p, captured) != make_pc(PAWN, us ^ 1)) return NO_SQ;
    const int left = captured - 1;
    const int right = captured + 1;
    bool candidate = false;
    if (on_board(left) && bb_piece_ref(p, left) == make_pc(PAWN, us)) candidate = true;
    if (on_board(right) && bb_piece_ref(p, right) == make_pc(PAWN, us)) candidate = true;
    return candidate ? p.ep : NO_SQ;
}
static uint64_t computeHash(const Position& p) {
    initZobrist();
    uint64_t k = 0;
    for (int sq = 0; sq < 128; ++sq)
        if (bb_piece_ref(p, sq)) k ^= zPiece[bb_piece_ref(p, sq)][sq];
    if (p.sideToMove == BLACK) k ^= zSide;
    k ^= zCastle[p.castle];
    k ^= zEp[hash_ep_square(p)];
    return k;
}
static int char_to_pc(char c) {
    int col = isupper((unsigned char)c) ? WHITE : BLACK;
    switch (tolower((unsigned char)c)) {
    case 'p': return make_pc(PAWN, col);
    case 'n': return make_pc(KNIGHT, col);
    case 'b': return make_pc(BISHOP, col);
    case 'r': return make_pc(ROOK, col);
    case 'q': return make_pc(QUEEN, col);
    case 'k': return make_pc(KING, col);
    default: return EMPTY;
    }
}
static bool parse_fen(Position& p, const char* fen) {
    clear_pos(p);
    int f = 0, r = 7;
    const char* s = fen;
    while (*s && *s != ' ') {
        if (*s == '/') {
            if (f != 8 || r == 0) return false;
            f = 0; --r;
        }
        else if (*s >= '1' && *s <= '8') {
            f += *s - '0';
            if (f > 8) return false;
        }
        else {
            int pc = char_to_pc(*s);
            if (!pc || !valid_file(f) || !valid_rank(r)) return false;
            bb_piece_ref(p, sq88(f, r)) = pc;
            ++f;
        }
        ++s;
    }
    if (f != 8 || r != 0 || *s != ' ') return false;
    ++s;
    if (*s == 'w') p.sideToMove = WHITE;
    else if (*s == 'b') p.sideToMove = BLACK;
    else return false;
    ++s;
    if (*s != ' ') return false;
    ++s;
    if (*s == '-') ++s;
    else {
        while (*s && *s != ' ') {
            if (*s == 'K') p.castle |= WK;
            else if (*s == 'Q') p.castle |= WQ;
            else if (*s == 'k') p.castle |= BK;
            else if (*s == 'q') p.castle |= BQ;
            else return false;
            ++s;
        }
    }
    if (*s != ' ') return false;
    ++s;
    if (*s == '-') { p.ep = NO_SQ; ++s; }
    else {
        if (*s < 'a' || *s > 'h') return false;
        int ef = *s++ - 'a';
        if (*s < '1' || *s > '8') return false;
        int er = *s++ - '1';
        p.ep = sq88(ef, er);
        if (!on_board(p.ep)) return false;
    }
    if (*s == ' ') {
        ++s;
        p.halfmove = 0;
        while (*s >= '0' && *s <= '9') p.halfmove = p.halfmove * 10 + (*s++ - '0');
        if (*s == ' ') {
            ++s;
            p.fullmove = 0;
            while (*s >= '0' && *s <= '9') p.fullmove = p.fullmove * 10 + (*s++ - '0');
            if (p.fullmove < 1) p.fullmove = 1;
        }
    }
    p.ksq[WHITE] = findKing(p, WHITE);
    p.ksq[BLACK] = findKing(p, BLACK);
    bb_rebuild(p);
    p.key = computeHash(p);
    return true;
}
static void set_start(Position& p) {
    parse_fen(p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}
// ----------------------------- MOVE GENERATOR --------------------------------
constexpr int MAX_MOVES = 256;
enum MoveFlag : int { F_CAPTURE = 1, F_PROMO = 2, F_CASTLE = 4, F_EP = 8, F_DOUBLE = 16 };
struct Move {
    uint8_t from;
    uint8_t to;
    uint8_t promo;
    uint8_t flags;
};
static bool move_equal(const Move& a, const Move& b) {
    return a.from == b.from && a.to == b.to && a.promo == b.promo && a.flags == b.flags;
}
struct MoveList {
    // build1.1 candidate: active entries are written by add(); only count is initialized.
    Move m[MAX_MOVES]{};
    int count = 0;
    void clear() { count = 0; }
    void add(int from, int to, int promo, int flags) {
        if (count >= MAX_MOVES) return;
        Move& x = m[count++];
        x.from = (uint8_t)from; x.to = (uint8_t)to;
        x.promo = (uint8_t)promo; x.flags = (uint8_t)flags;
    }
};
static const int KNIGHT_DIRS[8] = { 33, 31, 18, 14, -14, -18, -31, -33 };
static const int BISHOP_DIRS[4] = { 17, 15, -15, -17 };
static const int ROOK_DIRS[4] = { 16, 1, -1, -16 };
static const int KING_DIRS[8] = { 17, 16, 15, 1, -1, -15, -16, -17 };

// -----------------------------------------------------------------------------
// build1.1-bitboard stage 4: non-sliding attack tables.
// These tables are deliberately independent of occupancy. Sliding attacks are
// introduced in stage 5. The current 0x88 attack code remains the reference.
// -----------------------------------------------------------------------------
static Bitboard bb_pawn_attacks[COLOR_NB][64];
static Bitboard bb_knight_attacks[64];
static Bitboard bb_king_attacks[64];
static bool bb_attack_tables_ready = false;

static Bitboard bb_reference_step_attacks(int sq64, const int* dirs, int count) {
    Bitboard out = 0;
    const int from = sq88_from_bb_sq(sq64);
    for (int i = 0; i < count; ++i) {
        const int to = from + dirs[i];
        if (on_board(to)) out |= bb_square(bb_sq_from_sq88(to));
    }
    return out;
}

static Bitboard bb_reference_pawn_attacks(int color, int sq64) {
    const int from = sq88_from_bb_sq(sq64);
    const int dir = color == WHITE ? 16 : -16;
    Bitboard out = 0;
    for (int delta : { dir - 1, dir + 1 }) {
        const int to = from + delta;
        if (on_board(to)) out |= bb_square(bb_sq_from_sq88(to));
    }
    return out;
}

static void bb_init_attack_tables() {
    if (bb_attack_tables_ready) return;
    bb_attack_tables_ready = true;
    for (int sq64 = 0; sq64 < 64; ++sq64) {
        bb_pawn_attacks[WHITE][sq64] = bb_reference_pawn_attacks(WHITE, sq64);
        bb_pawn_attacks[BLACK][sq64] = bb_reference_pawn_attacks(BLACK, sq64);
        bb_knight_attacks[sq64] = bb_reference_step_attacks(
            sq64, KNIGHT_DIRS, 8);
        bb_king_attacks[sq64] = bb_reference_step_attacks(
            sq64, KING_DIRS, 8);
    }
}

static Bitboard bb_pawn_attack_mask(int color, int sq64) {
    bb_init_attack_tables();
    return bb_pawn_attacks[color][sq64];
}

static Bitboard bb_knight_attack_mask(int sq64) {
    bb_init_attack_tables();
    return bb_knight_attacks[sq64];
}

static Bitboard bb_king_attack_mask(int sq64) {
    bb_init_attack_tables();
    return bb_king_attacks[sq64];
}

// -----------------------------------------------------------------------------
// build1.1-bitboard stage 5: correct occupancy-aware sliding attacks.
// This remains the simple ray reference implementation used to build and
// exhaustively verify the fancy magic tables below against the existing 0x88
// isSquareAttacked() implementation.
// -----------------------------------------------------------------------------
static Bitboard bb_sliding_attacks_reference(int sq64, Bitboard occupied, bool bishop) {
    static const int bishop_df[4] = { 1, 1, -1, -1 };
    static const int bishop_dr[4] = { 1, -1, 1, -1 };
    static const int rook_df[4] = { 1, -1, 0, 0 };
    static const int rook_dr[4] = { 0, 0, 1, -1 };
    const int* dfs = bishop ? bishop_df : rook_df;
    const int* drs = bishop ? bishop_dr : rook_dr;
    const int count = 4;
    const int file = sq64 & 7;
    const int rank = sq64 >> 3;
    Bitboard attacks = 0;
    for (int i = 0; i < count; ++i) {
        int f = file + dfs[i];
        int r = rank + drs[i];
        while ((unsigned)f < 8 && (unsigned)r < 8) {
            const int to = r * 8 + f;
            const Bitboard bit = bb_square(to);
            attacks |= bit;
            if (occupied & bit) break;
            f += dfs[i];
            r += drs[i];
        }
    }
    return attacks;
}

// Fancy magic tables: the relevant occupancy bits for each square are hashed
// by a 64-bit multiply and a right shift into a compact per-square table.
// The reference ray implementation above remains available for initialization
// and exhaustive differential tests.
struct BBMagicEntry {
    Bitboard mask;
    Bitboard magic;
    uint32_t offset;
    uint8_t shift;
};

// generated magic constants; square = rank*8+file, a1=0
static const uint64_t bb_magic_bishop[64] = {
0x10102002004a1420ULL ,0x8020040400584008ULL ,
0x10510800811201c8ULL ,0x5204042080000088ULL ,
0x2204106880000002ULL ,0x1401042004000000ULL ,
0x0400880410042004ULL ,0x0028208200a02020ULL ,
0x1500241990010e00ULL ,0x8001200182020a40ULL ,
0x40004101030b0000ULL ,0x8002041042000100ULL ,
0x4010011041020038ULL ,0x0000010421044000ULL ,
0x1500210808020a00ULL ,0x8000088400880520ULL ,
0x0405004010040100ULL ,0x1005823210040108ULL ,
0x2708008102040011ULL ,0x4048200404009100ULL ,
0x0018104101400024ULL ,0x0003000601190101ULL ,
0x8004803108491000ULL ,0x8014241200820800ULL ,
0x0006e080100c3040ULL ,0x0501044a11041800ULL ,
0x9020300008004045ULL ,0x0894080000220040ULL ,
0x1001010083104000ULL ,0x5004030040900080ULL ,
0x000400422c012400ULL ,0x0002128698404812ULL ,
0x1010108404900440ULL ,0x0928021182084100ULL ,
0x2006080409020024ULL ,0x1010202020180080ULL ,
0xa010008200202200ULL ,0x2098015100019004ULL ,
0x0002041440810811ULL ,0x802a02020000b098ULL ,
0x0009015090004060ULL ,0x4000821082081001ULL ,
0x0100210040420800ULL ,0x0800004010488a00ULL ,
0x2000081104004040ULL ,0x4c8e029015000082ULL ,
0x0420340322224842ULL ,0x1298260043400210ULL ,
0x0000822802400008ULL ,0x00008a0101600000ULL ,
0x3040003412080021ULL ,0x3040290220884800ULL ,
0x4a1500401041004aULL ,0x8010200282020781ULL ,
0x0020203142209091ULL ,0x0070300600902110ULL ,
0x0040808800b62048ULL ,0x0000810400c44420ULL ,
0x00080400440c0441ULL ,0x8340080020840411ULL ,
0x0000000104208200ULL ,0x0000800810d00080ULL ,
0x0400530411080200ULL ,0x4040702400932244ULL
};
static const uint64_t bb_magic_rook[64] = {
0x1080004008801020ULL ,0x0840092002c03000ULL ,
0x1900200010400900ULL ,0x0880100008000480ULL ,
0x4200100420080200ULL ,0x8100020100080400ULL ,
0x0200040110886200ULL ,0x0200008040220411ULL ,
0x0404800084400220ULL ,0x0000401000402000ULL ,
0x0086001081220440ULL ,0x0408800800100280ULL ,
0x000a001201040820ULL ,0x8848800200840080ULL ,
0x4001000100040200ULL ,0x0442000102105084ULL ,
0x9080010020804100ULL ,0x0040404000201009ULL ,
0x0000808010002009ULL ,0x2200090021d00100ULL ,
0x0008008008040080ULL ,0x0004004002010040ULL ,
0x0011040008015042ULL ,0x00000a0001768104ULL ,
0x0000800080204009ULL ,0x2010004140002001ULL ,
0x9800200280100080ULL ,0x1000100080080080ULL ,
0x0442000a00049020ULL ,0x2100040080020080ULL ,
0x0800120400900148ULL ,0x0010040a00128541ULL ,
0x2800804000800030ULL ,0x1010002000400041ULL ,
0x4000200011004100ULL ,0x0610008410800800ULL ,
0x0400802402800800ULL ,0xc100020080800400ULL ,
0x0002000802000401ULL ,0x0182085882000401ULL ,
0x0220204000808000ULL ,0x2860100040024022ULL ,
0x0001002004110040ULL ,0x99101042000a0020ULL ,
0x0004080004008080ULL ,0x0010040002008080ULL ,
0x2012004881020004ULL ,0x8300842444820011ULL ,
0x0088403882010200ULL ,0x0820400080210100ULL ,
0x0110910040a00300ULL ,0x0801100280080480ULL ,
0x0242009008200600ULL ,0x1002000489500200ULL ,
0x0040800200010080ULL ,0x0091800041000080ULL ,
0x0000209300488001ULL ,0x04c1002414824001ULL ,
0x020020000b001041ULL ,0x7000100004200901ULL ,
0x8002002004100802ULL ,0x30010002084c0007ULL ,
0x0888221800813004ULL ,0x4000002840840112ULL
};

static BBMagicEntry bb_bishop_magic_info[64];
static BBMagicEntry bb_rook_magic_info[64];
static std::vector<Bitboard> bb_magic_attack_table;
static bool bb_magic_tables_ready = false;
static bool bb_magic_tables_valid = false;

static Bitboard bb_magic_relevant_mask(int sq64, bool bishop) {
    static const int bishop_df[4] = { 1, 1, -1, -1 };
    static const int bishop_dr[4] = { 1, -1, 1, -1 };
    static const int rook_df[4] = { 1, -1, 0, 0 };
    static const int rook_dr[4] = { 0, 0, 1, -1 };
    const int* dfs = bishop ? bishop_df : rook_df;
    const int* drs = bishop ? bishop_dr : rook_dr;
    const int file = sq64 & 7;
    const int rank = sq64 >> 3;
    Bitboard mask = 0;
    for (int i = 0; i < 4; ++i) {
        int f = file + dfs[i];
        int r = rank + drs[i];
        while ((unsigned)f < 8 && (unsigned)r < 8) {
            const int next_f = f + dfs[i];
            const int next_r = r + drs[i];
            // The outermost square cannot change the attack set: it is
            // attacked whether occupied or empty, so it is excluded from the
            // hash key. All inner ray squares remain relevant blockers.
            if ((unsigned)next_f >= 8 || (unsigned)next_r >= 8) break;
            mask |= bb_square(r * 8 + f);
            f = next_f;
            r = next_r;
        }
    }
    return mask;
}

static Bitboard bb_magic_subset_from_index(uint32_t index, Bitboard mask) {
    Bitboard subset = 0;
    while (mask) {
        const int sq64 = bb_ctz64(mask);
        mask &= mask - 1;
        if (index & 1u) subset |= bb_square(sq64);
        index >>= 1;
    }
    return subset;
}

static bool bb_init_magic_tables() {
    if (bb_magic_tables_ready) return bb_magic_tables_valid;
    bb_magic_tables_ready = true;
    bb_magic_tables_valid = true;
    bb_magic_attack_table.clear();
    bb_magic_attack_table.reserve(108000);

    auto init_piece = [&](bool bishop, BBMagicEntry* info,
                          const uint64_t* magics) {
        for (int sq64 = 0; sq64 < 64; ++sq64) {
            BBMagicEntry& entry = info[sq64];
            entry.mask = bb_magic_relevant_mask(sq64, bishop);
            const int bits = bb_popcount64(entry.mask);
            entry.shift = static_cast<uint8_t>(64 - bits);
            entry.magic = magics[sq64];
            entry.offset = static_cast<uint32_t>(bb_magic_attack_table.size());
            const uint32_t variations = 1u << bits;
            bb_magic_attack_table.resize(
                bb_magic_attack_table.size() + variations, 0);
            std::vector<uint8_t> assigned(variations, 0);
            for (uint32_t i = 0; i < variations; ++i) {
                const Bitboard occupied =
                    bb_magic_subset_from_index(i, entry.mask);
                const uint32_t magic_index = static_cast<uint32_t>(
                    ((occupied * entry.magic) >> entry.shift));
                const Bitboard attacks =
                    bb_sliding_attacks_reference(sq64, occupied, bishop);
                Bitboard& slot =
                    bb_magic_attack_table[entry.offset + magic_index];
                if (assigned[magic_index]) {
                    if (slot != attacks) {
                        bb_magic_tables_valid = false;
                        return;
                    }
                }
                else {
                    slot = attacks;
                    assigned[magic_index] = 1;
                }
            }
        }
    };

    init_piece(true, bb_bishop_magic_info, bb_magic_bishop);
    if (bb_magic_tables_valid)
        init_piece(false, bb_rook_magic_info, bb_magic_rook);
    if (!bb_magic_tables_valid) {
        bb_magic_attack_table.clear();
        std::printf("info string magic tables invalid, using reference rays\n");
    }
    return bb_magic_tables_valid;
}

static inline Bitboard bb_magic_lookup(const BBMagicEntry& entry,
                                       Bitboard occupied) {
    const uint32_t index = static_cast<uint32_t>(
        (((occupied & entry.mask) * entry.magic) >> entry.shift));
    return bb_magic_attack_table[entry.offset + index];
}

static Bitboard bb_bishop_attack_mask(int sq64, Bitboard occupied) {
    if (!bb_init_magic_tables())
        return bb_sliding_attacks_reference(sq64, occupied, true);
    return bb_magic_lookup(bb_bishop_magic_info[sq64], occupied);
}

static Bitboard bb_rook_attack_mask(int sq64, Bitboard occupied) {
    if (!bb_init_magic_tables())
        return bb_sliding_attacks_reference(sq64, occupied, false);
    return bb_magic_lookup(bb_rook_magic_info[sq64], occupied);
}

static Bitboard bb_queen_attack_mask(int sq64, Bitboard occupied) {
    return bb_bishop_attack_mask(sq64, occupied) |
        bb_rook_attack_mask(sq64, occupied);
}

static void add_promos(MoveList& ml, int from, int to, int flags) {
    static const int pr[4] = { QUEEN, ROOK, BISHOP, KNIGHT };
    for (int i = 0; i < 4; ++i) ml.add(from, to, pr[i], flags | F_PROMO);
}
static void gen_steps(const Position& p, int sq, int us, const int* d, int n, MoveList& ml) {
    for (int i = 0; i < n; ++i) {
        int to = sq + d[i];
        if (!on_board(to)) continue;
        int pc = bb_piece_ref(p, to);
        if (!pc) ml.add(sq, to, 0, 0);
        else if (color_of(pc) != us) ml.add(sq, to, 0, F_CAPTURE);
    }
}
static void gen_slides(const Position& p, int sq, int us, const int* d, int n, MoveList& ml) {
    for (int i = 0; i < n; ++i) {
        for (int to = sq + d[i]; on_board(to); to += d[i]) {
            int pc = bb_piece_ref(p, to);
            if (!pc) { ml.add(sq, to, 0, 0); continue; }
            if (color_of(pc) != us) ml.add(sq, to, 0, F_CAPTURE);
            break;
        }
    }
}
static void gen_pawn(const Position& p, int sq, int us, MoveList& ml) {
    const int dir = (us == WHITE) ? 16 : -16;
    const int startRank = (us == WHITE) ? 1 : 6;
    const int promoRank = (us == WHITE) ? 7 : 0;
    int one = sq + dir;
    if (on_board(one) && bb_piece_ref(p, one) == EMPTY) {
        if (rank_of(one) == promoRank) add_promos(ml, sq, one, 0);
        else {
            ml.add(sq, one, 0, 0);
            int two = one + dir;
            if (rank_of(sq) == startRank && on_board(two) && bb_piece_ref(p, two) == EMPTY)
                ml.add(sq, two, 0, F_DOUBLE);
        }
    }
    const int caps[2] = { dir - 1, dir + 1 };
    for (int i = 0; i < 2; ++i) {
        int to = sq + caps[i];
        if (!on_board(to)) continue;
        int pc = bb_piece_ref(p, to);
        if (pc) {
            if (color_of(pc) == us) continue;
            if (rank_of(to) == promoRank) add_promos(ml, sq, to, F_CAPTURE);
            else ml.add(sq, to, 0, F_CAPTURE);
        }
        else if (p.ep != NO_SQ && to == p.ep) {
            // build0.3: an ep square from a malformed/hand-written FEN (or any
            // stale raw p.ep) must not produce a phantom en-passant capture
            // when there is no actual pawn to remove. We require:
            //   - the ep square sits on the correct rank for `us` to capture
            //     onto (rank 5 for White, rank 2 for Black);
            //   - the square behind it (relative to `us`) actually holds an
            //     enemy pawn that would be captured.
            const bool rankOk = (us == WHITE) ? (rank_of(p.ep) == 5) : (rank_of(p.ep) == 2);
            if (rankOk) {
                int cs = to + (us == WHITE ? -16 : 16);
                if (on_board(cs) && bb_piece_ref(p, cs) == make_pc(PAWN, us ^ 1)) {
                    ml.add(sq, to, 0, F_CAPTURE | F_EP);
                }
            }
        }
    }
}
static void gen_castle(const Position& p, int us, MoveList& ml) {
    if (us == WHITE) {
        constexpr int e1 = sq88(4, 0);
        if (bb_piece_ref(p, e1) != make_pc(KING, WHITE)) return;
        if ((p.castle & WK) && bb_piece_ref(p, sq88(7, 0)) == make_pc(ROOK, WHITE)
            && !bb_piece_ref(p, sq88(5, 0)) && !bb_piece_ref(p, sq88(6, 0)))
            ml.add(e1, sq88(6, 0), 0, F_CASTLE);
        if ((p.castle & WQ) && bb_piece_ref(p, sq88(0, 0)) == make_pc(ROOK, WHITE)
            && !bb_piece_ref(p, sq88(1, 0)) && !bb_piece_ref(p, sq88(2, 0)) && !bb_piece_ref(p, sq88(3, 0)))
            ml.add(e1, sq88(2, 0), 0, F_CASTLE);
    }
    else {
        constexpr int e8 = sq88(4, 7);
        if (bb_piece_ref(p, e8) != make_pc(KING, BLACK)) return;
        if ((p.castle & BK) && bb_piece_ref(p, sq88(7, 7)) == make_pc(ROOK, BLACK)
            && !bb_piece_ref(p, sq88(5, 7)) && !bb_piece_ref(p, sq88(6, 7)))
            ml.add(e8, sq88(6, 7), 0, F_CASTLE);
        if ((p.castle & BQ) && bb_piece_ref(p, sq88(0, 7)) == make_pc(ROOK, BLACK)
            && !bb_piece_ref(p, sq88(1, 7)) && !bb_piece_ref(p, sq88(2, 7)) && !bb_piece_ref(p, sq88(3, 7)))
            ml.add(e8, sq88(2, 7), 0, F_CASTLE);
    }
}
static void generateMoves_0x88(const Position& p, MoveList& ml) {
    ml.clear();
    const int us = p.sideToMove;
    for (int r = 0; r < 8; ++r) for (int f = 0; f < 8; ++f) {
        int sq = sq88(f, r);
        int pc = bb_piece_ref(p, sq);
        if (!pc || color_of(pc) != us) continue;
        switch (piece_of(pc)) {
        case PAWN:   gen_pawn(p, sq, us, ml); break;
        case KNIGHT: gen_steps(p, sq, us, KNIGHT_DIRS, 8, ml); break;
        case BISHOP: gen_slides(p, sq, us, BISHOP_DIRS, 4, ml); break;
        case ROOK:   gen_slides(p, sq, us, ROOK_DIRS, 4, ml); break;
        case QUEEN:  gen_slides(p, sq, us, BISHOP_DIRS, 4, ml);
            gen_slides(p, sq, us, ROOK_DIRS, 4, ml); break;
        case KING:   gen_steps(p, sq, us, KING_DIRS, 8, ml); break;
        default: break;
        }
    }
    gen_castle(p, us, ml);
}
static int generateMoves_0x88(const Position& p, Move* out) {
    MoveList ml; generateMoves_0x88(p, ml);
    for (int i = 0; i < ml.count; ++i) out[i] = ml.m[i];
    return ml.count;
}
// -----------------------------------------------------------------------------
// build1.1-bitboard stage 7: pseudo-legal move generation.
// The old 0x88 generator remains available as generateMoves_0x88() for
// differential testing. The active generateMoves() path uses the bitboard
// mirror for source iteration, occupancy, targets, and castling rights.
// -----------------------------------------------------------------------------
static int bb_pop_lsb(Bitboard& bits) {
    const int sq = bb_ctz64(bits);
    bits &= bits - 1;
    return sq;
}

static void bb_add_step_moves(const Position& p, int sq64, int piece,
    MoveList& ml) {
    const int us = p.sideToMove;
    const Bitboard own = p.bbOccupied[us];
    const Bitboard enemy = p.bbOccupied[us ^ 1];
    Bitboard targets = (piece == KNIGHT)
        ? bb_knight_attack_mask(sq64)
        : bb_king_attack_mask(sq64);
    targets &= ~own;
    while (targets) {
        const int to64 = bb_pop_lsb(targets);
        const int flags = (enemy & bb_square(to64)) ? F_CAPTURE : 0;
        ml.add(sq88_from_bb_sq(sq64), sq88_from_bb_sq(to64), 0, flags);
    }
}

static void bb_add_slider_moves(const Position& p, int sq64, int piece,
    MoveList& ml) {
    const int us = p.sideToMove;
    const Bitboard own = p.bbOccupied[us];
    const Bitboard enemy = p.bbOccupied[us ^ 1];
    const Bitboard occupied = p.bbOccupiedAll;
    Bitboard targets = 0;
    if (piece == BISHOP) targets = bb_bishop_attack_mask(sq64, occupied);
    else if (piece == ROOK) targets = bb_rook_attack_mask(sq64, occupied);
    else {
        // Preserve the old generator's queen ordering: diagonals first,
        // orthogonals second.
        targets = bb_bishop_attack_mask(sq64, occupied);
        Bitboard rook_targets = bb_rook_attack_mask(sq64, occupied);
        targets &= ~own;
        while (targets) {
            const int to64 = bb_pop_lsb(targets);
            const int flags = (enemy & bb_square(to64)) ? F_CAPTURE : 0;
            ml.add(sq88_from_bb_sq(sq64), sq88_from_bb_sq(to64), 0, flags);
        }
        rook_targets &= ~own;
        while (rook_targets) {
            const int to64 = bb_pop_lsb(rook_targets);
            const int flags = (enemy & bb_square(to64)) ? F_CAPTURE : 0;
            ml.add(sq88_from_bb_sq(sq64), sq88_from_bb_sq(to64), 0, flags);
        }
        return;
    }
    targets &= ~own;
    while (targets) {
        const int to64 = bb_pop_lsb(targets);
        const int flags = (enemy & bb_square(to64)) ? F_CAPTURE : 0;
        ml.add(sq88_from_bb_sq(sq64), sq88_from_bb_sq(to64), 0, flags);
    }
}

static void bb_gen_pawns(const Position& p, MoveList& ml) {
    const int us = p.sideToMove;
    const int dir = us == WHITE ? 8 : -8;
    const int start_rank = us == WHITE ? 1 : 6;
    const int promo_rank = us == WHITE ? 7 : 0;
    const Bitboard occupied = p.bbOccupiedAll;
    const Bitboard enemy = p.bbOccupied[us ^ 1];
    const Bitboard enemy_pawns = p.bbPieces[us ^ 1][PAWN];
    Bitboard pawns = p.bbPieces[us][PAWN];
    while (pawns) {
        const int from64 = bb_pop_lsb(pawns);
        const int from88 = sq88_from_bb_sq(from64);
        const int from_rank = from64 >> 3;
        const int one64 = from64 + dir;
        if ((unsigned)one64 < 64 && !(occupied & bb_square(one64))) {
            const int one_rank = one64 >> 3;
            if (one_rank == promo_rank) {
                add_promos(ml, from88, sq88_from_bb_sq(one64), 0);
            }
            else {
                ml.add(from88, sq88_from_bb_sq(one64), 0, 0);
                const int two64 = from64 + 2 * dir;
                if (from_rank == start_rank && (unsigned)two64 < 64 &&
                    !(occupied & bb_square(two64))) {
                    ml.add(from88, sq88_from_bb_sq(two64), 0, F_DOUBLE);
                }
            }
        }

        Bitboard captures = bb_pawn_attack_mask(us, from64) & enemy;
        while (captures) {
            const int to64 = bb_pop_lsb(captures);
            const int to_rank = to64 >> 3;
            if (to_rank == promo_rank)
                add_promos(ml, from88, sq88_from_bb_sq(to64), F_CAPTURE);
            else
                ml.add(from88, sq88_from_bb_sq(to64), 0, F_CAPTURE);
        }

        if (p.ep != NO_SQ && on_board(p.ep)) {
            const int ep64 = bb_sq_from_sq88(p.ep);
            const bool rank_ok = us == WHITE ? ((ep64 >> 3) == 5) : ((ep64 >> 3) == 2);
            const int captured64 = ep64 + (us == WHITE ? -8 : 8);
            if (rank_ok && (unsigned)captured64 < 64 &&
                (enemy_pawns & bb_square(captured64)) &&
                (bb_pawn_attack_mask(us, from64) & bb_square(ep64)) &&
                !(occupied & bb_square(ep64))) {
                ml.add(from88, p.ep, 0, F_CAPTURE | F_EP);
            }
        }
    }
}

static void bb_gen_castle(const Position& p, MoveList& ml) {
    const int us = p.sideToMove;
    const int rank = us == WHITE ? 0 : 7;
    const int e = sq88(4, rank);
    const int king64 = rank * 8 + 4;
    if (!(p.bbPieces[us][KING] & bb_square(king64))) return;
    const Bitboard occupied = p.bbOccupiedAll;
    const Bitboard rook = p.bbPieces[us][ROOK];
    const int king_side = us == WHITE ? WK : BK;
    const int queen_side = us == WHITE ? WQ : BQ;
    const int rook_king64 = rank * 8 + 7;
    const int rook_queen64 = rank * 8;
    if ((p.castle & king_side) && (rook & bb_square(rook_king64)) &&
        !(occupied & (bb_square(rank * 8 + 5) | bb_square(rank * 8 + 6)))) {
        ml.add(e, sq88(6, rank), 0, F_CASTLE);
    }
    if ((p.castle & queen_side) && (rook & bb_square(rook_queen64)) &&
        !(occupied & (bb_square(rank * 8 + 1) | bb_square(rank * 8 + 2) |
                      bb_square(rank * 8 + 3)))) {
        ml.add(e, sq88(2, rank), 0, F_CASTLE);
    }
}

static void generateMoves(const Position& p, MoveList& ml) {
    ml.clear();
    bb_init_attack_tables();
    bb_gen_pawns(p, ml);
    const int us = p.sideToMove;
    Bitboard pieces = p.bbPieces[us][KNIGHT];
    while (pieces) {
        const int from64 = bb_pop_lsb(pieces);
        bb_add_step_moves(p, from64, KNIGHT, ml);
    }
    pieces = p.bbPieces[us][BISHOP];
    while (pieces) {
        const int from64 = bb_pop_lsb(pieces);
        bb_add_slider_moves(p, from64, BISHOP, ml);
    }
    pieces = p.bbPieces[us][ROOK];
    while (pieces) {
        const int from64 = bb_pop_lsb(pieces);
        bb_add_slider_moves(p, from64, ROOK, ml);
    }
    pieces = p.bbPieces[us][QUEEN];
    while (pieces) {
        const int from64 = bb_pop_lsb(pieces);
        bb_add_slider_moves(p, from64, QUEEN, ml);
    }
    pieces = p.bbPieces[us][KING];
    while (pieces) {
        const int from64 = bb_pop_lsb(pieces);
        bb_add_step_moves(p, from64, KING, ml);
    }
    bb_gen_castle(p, ml);
}

static int generateMoves(const Position& p, Move* out) {
    MoveList ml;
    generateMoves(p, ml);
    for (int i = 0; i < ml.count; ++i) out[i] = ml.m[i];
    return ml.count;
}

// ------------------------------- MAKE / UNMAKE -------------------------------
struct Undo {
    int captured;
    int castle;
    int ep;
    int halfmove;
    int fullmove;
    int ksq;
    uint64_t key;
};
static int castle_mask(int sq) {
    int x = WK | WQ | BK | BQ;
    if (sq == sq88(4, 0) || sq == sq88(7, 0)) x &= ~WK;
    if (sq == sq88(4, 0) || sq == sq88(0, 0)) x &= ~WQ;
    if (sq == sq88(4, 7) || sq == sq88(7, 7)) x &= ~BK;
    if (sq == sq88(4, 7) || sq == sq88(0, 7)) x &= ~BQ;
    return x;
}
// =============================================================================
// REPETITION DETECTION — explicit, caller-owned stack (NOT global, NOT tied to
// makeMove/unmakeMove).
//
// Раньше это было глобальное состояние (position_history[]/history_ply),
// которое makeMove/unmakeMove трогали неявно. Из-за этого:
//   - perft/bench были вынуждены руками save/restore чужой глобал,
//   - несколько независимых поисков (будущий Lazy SMP / self-play) не могли
//     сосуществовать — все писали в один и тот же массив,
//   - при переполнении HISTORY_SIZE счётчик и реальный стек расходились.
//
// Теперь makeMove/unmakeMove абсолютно чистые (только board/key/castle/ep),
// а история повторов — обычный объект, которым явно владеет вызывающий код
// (UCI-слой хранит реальную партию, search() получает свою копию для дерева
// поиска).
// =============================================================================
struct RepetitionStack {
    static constexpr int CAPACITY = 1024;
    uint64_t keys[CAPACITY]{};
    int count = 0;
    void push(uint64_t k) { if (count < CAPACITY) keys[count++] = k; }
    void pop() { if (count > 0) --count; }
};
void makeMove(Position& p, const Move& m, Undo& u) {
    const int from = m.from, to = m.to, us = p.sideToMove;
    const int pc = bb_piece_ref(p, from);
    u.captured = EMPTY;
    u.castle = p.castle;
    u.ep = p.ep;
    u.halfmove = p.halfmove;
    u.fullmove = p.fullmove;
    u.ksq = p.ksq[us];
    u.key = p.key;
    // build1.1-bitboard stage 9: update the authoritative bitboards and the
    // compact 64-square piece lookup together.
    bb_remove_piece(p, pc, from);

    // build0.3: EP hashing is handled separately from the rest of the
    // incremental key update via hash_ep_square(), which normalizes away any
    // ep square that could not actually be captured. We first strip the OLD
    // (already-normalized) ep component from the key while `p` still fully
    // describes the pre-move position...
    const int old_ep_component = hash_ep_square(p);
    p.key ^= zEp[old_ep_component];
    p.key ^= zSide ^ zCastle[p.castle] ^ zPiece[pc][from];
    p.halfmove = (piece_of(pc) == PAWN || (m.flags & F_CAPTURE)) ? 0 : p.halfmove + 1;
    if (m.flags & F_EP) {
        int cs = to + (us == WHITE ? -16 : 16);
        u.captured = bb_piece_ref(p, cs);
        p.key ^= zPiece[u.captured][cs];
        bb_remove_piece(p, u.captured, cs);
        bb_piece_ref(p, cs) = EMPTY;
    }
    else if (m.flags & F_CAPTURE) {
        u.captured = bb_piece_ref(p, to);
        p.key ^= zPiece[u.captured][to];
        bb_remove_piece(p, u.captured, to);
    }
    bb_piece_ref(p, from) = EMPTY;
    bb_piece_ref(p, to) = (m.flags & F_PROMO) ? make_pc(m.promo, us) : pc;
    bb_add_piece(p, bb_piece_ref(p, to), to);
    if (piece_of(pc) == KING) p.ksq[us] = to;
    p.key ^= zPiece[bb_piece_ref(p, to)][to];
    if (m.flags & F_CASTLE) {
        int r = rank_of(to);
        int ks = file_of(to) == 6;
        int rf = sq88(ks ? 7 : 0, r), rt = sq88(ks ? 5 : 3, r);
        p.key ^= zPiece[bb_piece_ref(p, rf)][rf] ^ zPiece[bb_piece_ref(p, rf)][rt];
        bb_move_piece(p, bb_piece_ref(p, rf), rf, rt);
        bb_piece_ref(p, rt) = bb_piece_ref(p, rf);
        bb_piece_ref(p, rf) = EMPTY;
    }
    p.castle &= castle_mask(from) & castle_mask(to);
    p.key ^= zCastle[p.castle];
    p.ep = (m.flags & F_DOUBLE) ? (from + (us == WHITE ? 16 : -16)) : NO_SQ;
    if (us == BLACK) ++p.fullmove;
    p.sideToMove ^= 1;
    // ...then, once the position (board + side + raw ep) fully describes the
    // POST-move position, add back the NEW normalized ep component. This
    // keeps p.key == computeHash(p) after every makeMove() call.
    const int new_ep_component = hash_ep_square(p);
    p.key ^= zEp[new_ep_component];
}
void unmakeMove(Position& p, const Move& m, const Undo& u) {
    p.sideToMove ^= 1;
    p.key = u.key;
    const int from = m.from, to = m.to, us = p.sideToMove;
    p.castle = u.castle;
    p.ep = u.ep;
    p.halfmove = u.halfmove;
    p.fullmove = u.fullmove;
    if (m.flags & F_CASTLE) {
        int r = rank_of(to);
        int ks = file_of(to) == 6;
        int rf = sq88(ks ? 7 : 0, r), rt = sq88(ks ? 5 : 3, r);
        bb_move_piece(p, bb_piece_ref(p, rt), rt, rf);
        bb_piece_ref(p, rf) = bb_piece_ref(p, rt);
        bb_piece_ref(p, rt) = EMPTY;
    }
    const int moved_pc = bb_piece_ref(p, to);
    bb_remove_piece(p, moved_pc, to);
    bb_piece_ref(p, from) = (m.flags & F_PROMO) ? make_pc(PAWN, us) : moved_pc;
    bb_add_piece(p, bb_piece_ref(p, from), from);
    if (m.flags & F_EP) {
        bb_piece_ref(p, to) = EMPTY;
        bb_piece_ref(p, to + (us == WHITE ? -16 : 16)) = u.captured;
        bb_add_piece(p, u.captured, to + (us == WHITE ? -16 : 16));
    }
    else {
        bb_piece_ref(p, to) = u.captured;
        bb_add_piece(p, u.captured, to);
    }
    p.ksq[us] = u.ksq;
}
// -------------------------- CHECK / LEGAL MOVES -----------------------------
bool isSquareAttacked_0x88(const Position& p, int sq, int byColor) {
    int pdir = (byColor == WHITE) ? -16 : 16;
    int pd[2] = { pdir - 1, pdir + 1 };
    for (int i = 0; i < 2; ++i) {
        int s = sq + pd[i];
        if (on_board(s) && bb_piece_ref(p, s) == make_pc(PAWN, byColor)) return true;
    }
    for (int d : KNIGHT_DIRS) {
        int s = sq + d;
        if (on_board(s) && bb_piece_ref(p, s) == make_pc(KNIGHT, byColor)) return true;
    }
    for (int d : KING_DIRS) {
        int s = sq + d;
        if (on_board(s) && bb_piece_ref(p, s) == make_pc(KING, byColor)) return true;
    }
    for (int d : BISHOP_DIRS) {
        for (int s = sq + d; on_board(s); s += d) {
            int pc = bb_piece_ref(p, s);
            if (!pc) continue;
            if (color_of(pc) == byColor && (piece_of(pc) == BISHOP || piece_of(pc) == QUEEN)) return true;
            break;
        }
    }
    for (int d : ROOK_DIRS) {
        for (int s = sq + d; on_board(s); s += d) {
            int pc = bb_piece_ref(p, s);
            if (!pc) continue;
            if (color_of(pc) == byColor && (piece_of(pc) == ROOK || piece_of(pc) == QUEEN)) return true;
            break;
        }
    }
    return false;
}
static int bb_first_candidate_on_dirs(Bitboard candidates, int target64,
    const int* dirs, int count) {
    const int target88 = sq88_from_bb_sq(target64);
    for (int i = 0; i < count; ++i) {
        for (int sq = target88 + dirs[i]; on_board(sq); sq += dirs[i]) {
            const Bitboard bit = bb_square(bb_sq_from_sq88(sq));
            if (candidates & bit) return sq;
        }
    }
    return NO_SQ;
}

static int bb_first_pawn_attacker(Bitboard candidates, int target64, int side) {
    const int target88 = sq88_from_bb_sq(target64);
    const int fwd = side == WHITE ? 16 : -16;
    for (int df : { -1, 1 }) {
        const int sq = target88 - fwd + df;
        if (on_board(sq) && (candidates & bb_square(bb_sq_from_sq88(sq))))
            return sq;
    }
    return NO_SQ;
}

static bool isSquareAttacked_bitboard(const Position& p, int sq, int byColor) {
    if (!on_board(sq)) return false;
    const int target64 = bb_sq_from_sq88(sq);
    const Bitboard target = bb_square(target64);
    const Bitboard removed = 0;
    const Bitboard live = ~removed;

    Bitboard candidates = p.bbPieces[byColor][PAWN] & live;
    if (bb_first_pawn_attacker(candidates, target64, byColor) != NO_SQ)
        return true;

    candidates = p.bbPieces[byColor][KNIGHT] & live &
        bb_knight_attack_mask(target64);
    if (candidates) return true;
    candidates = p.bbPieces[byColor][KING] & live &
        bb_king_attack_mask(target64);
    if (candidates) return true;

    const Bitboard occupied = p.bbOccupiedAll & live;
    const Bitboard bishop_rays = bb_bishop_attack_mask(target64, occupied);
    const Bitboard rook_rays = bb_rook_attack_mask(target64, occupied);
    if (bishop_rays & (p.bbPieces[byColor][BISHOP] | p.bbPieces[byColor][QUEEN]))
        return true;
    if (rook_rays & (p.bbPieces[byColor][ROOK] | p.bbPieces[byColor][QUEEN]))
        return true;
    return false;
}

bool isSquareAttacked(const Position& p, int sq, int byColor) {
    return isSquareAttacked_bitboard(p, sq, byColor);
}

int findKing(const Position& p, int color) {
    int target = make_pc(KING, color);
    for (int sq = 0; sq < 128; ++sq)
        if (on_board(sq) && bb_piece_ref(p, sq) == target) return sq;
    return NO_SQ;
}
bool inCheck(const Position& p, int color) {
    const int ksq = p.ksq[color];
    return ksq != NO_SQ && isSquareAttacked(p, ksq, color ^ 1);
}

static bool inCheck_0x88(const Position& p, int color) {
    const int ksq = p.ksq[color];
    return ksq != NO_SQ && isSquareAttacked_0x88(p, ksq, color ^ 1);
}

// Preserved legal reference: old pseudo generator + old attack detector.
static void generateLegalMoves_0x88(const Position& p, MoveList& ml) {
    MoveList pseudo;
    generateMoves_0x88(p, pseudo);
    ml.clear();
    Position pos = p;
    const int us = pos.sideToMove;
    for (int i = 0; i < pseudo.count; ++i) {
        const Move& m = pseudo.m[i];
        if (m.flags & F_CASTLE) {
            if (inCheck_0x88(pos, us)) continue;
            if (isSquareAttacked_0x88(pos, (m.from + m.to) / 2, us ^ 1)) continue;
            if (isSquareAttacked_0x88(pos, m.to, us ^ 1)) continue;
        }
        Undo u;
        makeMove(pos, m, u);
        if (!inCheck_0x88(pos, us)) ml.add(m.from, m.to, m.promo, m.flags);
        unmakeMove(pos, m, u);
    }
}

// Active legal filtering: bitboard pseudo moves, bitboard attack tests, and
// the existing make/unmake state transition. Pins and discovered checks are
// rejected by testing the mover's king after every candidate move. Castling
// checks the origin, transit, and destination squares before making the move.
void generateLegalMoves(const Position& p, MoveList& ml) {
    MoveList pseudo;
    generateMoves(p, pseudo);
    ml.clear();
    Position pos = p;
    const int us = pos.sideToMove;
    for (int i = 0; i < pseudo.count; ++i) {
        const Move& m = pseudo.m[i];
        if (m.flags & F_CASTLE) {
            if (inCheck(pos, us)) continue;
            if (isSquareAttacked(pos, (m.from + m.to) / 2, us ^ 1)) continue;
            if (isSquareAttacked(pos, m.to, us ^ 1)) continue;
        }
        Undo u;
        makeMove(pos, m, u);
        if (!inCheck(pos, us)) ml.add(m.from, m.to, m.promo, m.flags);
        unmakeMove(pos, m, u);
    }
}
int generateLegalMoves(const Position& p, Move* out) {
    MoveList ml;
    generateLegalMoves(p, ml);
    if (out) for (int i = 0; i < ml.count; ++i) out[i] = ml.m[i];
    return ml.count;
}
// -------------------------------- SELF-TESTS ---------------------------------
static int g_pass = 0, g_fail = 0;
static void check(bool ok, const char* name) {
    if (ok) ++g_pass; else { ++g_fail; std::printf("FAIL: %s\n", name); }
}
static int sq_of(const char* s) { return sq88(s[0] - 'a', s[1] - '1'); }
static void move_str(const Move& m, char* b) {
    b[0] = (char)('a' + file_of(m.from));
    b[1] = (char)('1' + rank_of(m.from));
    b[2] = (char)('a' + file_of(m.to));
    b[3] = (char)('1' + rank_of(m.to));
    int n = 4;
    if (m.flags & F_PROMO) {
        switch (m.promo) {
        case KNIGHT:
            b[n++] = 'n';
            break;
        case BISHOP:
            b[n++] = 'b';
            break;
        case ROOK:
            b[n++] = 'r';
            break;
        case QUEEN:
            b[n++] = 'q';
            break;
        }
    }
    b[n] = 0;
}
// ------------------------------- PERFT TESTER --------------------------------
// Perft считает количество легальных позиций на глубине depth.
// Это НЕ поиск и НЕ оценка. Нужен только для проверки генератора ходов,
// make/unmake, рокировки, en-passant и превращений.
//
// perft больше НЕ трогает никакой глобал: makeMove/unmakeMove чистые.
static uint64_t perft(Position& pos, int depth) {
    if (depth <= 0)
        return 1ULL;
    MoveList ml;
    generateLegalMoves(pos, ml);
    if (depth == 1)
        return (uint64_t)ml.count;
    uint64_t nodes = 0;
    for (int i = 0; i < ml.count; ++i) {
        Undo u;
        makeMove(pos, ml.m[i], u);
        nodes += perft(pos, depth - 1);
        unmakeMove(pos, ml.m[i], u);
    }
    return nodes;
}
// Divide показывает вклад каждого хода первого слоя.
// Очень удобно, когда общий perft не совпал: видно, на каком ходе ошибка.
static uint64_t perft_divide(Position& pos, int depth) {
    if (depth <= 0) {
        std::printf("info string nodes 1\n");
        std::fflush(stdout);
        return 1ULL;
    }
    MoveList ml;
    generateLegalMoves(pos, ml);
    uint64_t total_nodes = 0;
    for (int i = 0; i < ml.count; ++i) {
        Undo u;
        uint64_t child_nodes;
        if (depth == 1) {
            child_nodes = 1ULL;
        }
        else {
            makeMove(pos, ml.m[i], u);
            child_nodes = perft(pos, depth - 1);
            unmakeMove(pos, ml.m[i], u);
        }
        char buf[8];
        move_str(ml.m[i], buf);
        std::printf("info string %s %llu\n", buf, (unsigned long long)child_nodes);
        total_nodes += child_nodes;
    }
    std::printf("info string total %llu\n", (unsigned long long)total_nodes);
    std::fflush(stdout);
    return total_nodes;
}
static bool perft_check(const char* name, const char* fen, int depth, uint64_t expected) {
    Position p;
    if (!parse_fen(p, fen)) {
        std::printf("info string PERFT FAIL %s depth %d bad fen\n", name, depth);
        std::fflush(stdout);
        return false;
    }
    uint64_t got = perft(p, depth);
    bool ok = (got == expected);
    std::printf("info string PERFT %s depth %d nodes %llu expected %llu %s\n",
        name,
        depth,
        (unsigned long long)got,
        (unsigned long long)expected,
        ok ? "OK" : "FAIL");
    std::fflush(stdout);
    return ok;
}
static void run_perft_tests() {
    int pass = 0;
    int fail = 0;
    const char* STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const char* KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    auto one = [&](const char* name, const char* fen, int depth, uint64_t expected) {
        if (perft_check(name, fen, depth, expected)) ++pass;
        else ++fail;
        };
    one("startpos", STARTPOS, 1, 20ULL);
    one("startpos", STARTPOS, 2, 400ULL);
    one("startpos", STARTPOS, 3, 8902ULL);
    one("startpos", STARTPOS, 4, 197281ULL);
    one("startpos", STARTPOS, 5, 4865609ULL);
    // Kiwipete: классическая позиция для проверки рокировок, взятий и сложных ходов.
    one("kiwipete", KIWIPETE, 1, 48ULL);
    one("kiwipete", KIWIPETE, 2, 2039ULL);
    one("kiwipete", KIWIPETE, 3, 97862ULL);
    one("kiwipete", KIWIPETE, 4, 4085603ULL);
    std::printf("info string PERFT TESTS DONE pass %d fail %d\n", pass, fail);
    std::fflush(stdout);
}
static int bb_move_key(const Move& m) {
    return int(m.from) | (int(m.to) << 8) |
        (int(m.promo) << 16) | (int(m.flags) << 20);
}

static bool bb_move_lists_equal(const MoveList& old_list, const MoveList& new_list) {
    if (old_list.count != new_list.count) return false;
    std::vector<int> old_keys, new_keys;
    old_keys.reserve(old_list.count);
    new_keys.reserve(new_list.count);
    for (int i = 0; i < old_list.count; ++i)
        old_keys.push_back(bb_move_key(old_list.m[i]));
    for (int i = 0; i < new_list.count; ++i)
        new_keys.push_back(bb_move_key(new_list.m[i]));
    std::sort(old_keys.begin(), old_keys.end());
    std::sort(new_keys.begin(), new_keys.end());
    return old_keys == new_keys;
}

static bool bb_position_core_equal(const Position& a, const Position& b) {
    return std::memcmp(a.piece, b.piece, sizeof a.piece) == 0 &&
        a.sideToMove == b.sideToMove && a.castle == b.castle && a.ep == b.ep &&
        a.halfmove == b.halfmove && a.fullmove == b.fullmove &&
        a.ksq[WHITE] == b.ksq[WHITE] && a.ksq[BLACK] == b.ksq[BLACK] &&
        a.key == b.key &&
        std::memcmp(a.bbPieces, b.bbPieces, sizeof a.bbPieces) == 0 &&
        std::memcmp(a.bbOccupied, b.bbOccupied, sizeof a.bbOccupied) == 0 &&
        a.bbOccupiedAll == b.bbOccupiedAll;
}

static bool bb_walk_make_unmake(Position& p, int depth) {
    if (!bb_state_matches(p)) return false;
    if (depth == 0) return true;
    MoveList ml;
    generateLegalMoves(p, ml);
    for (int i = 0; i < ml.count; ++i) {
        Position before = p;
        Undo u;
        makeMove(p, ml.m[i], u);
        if (!bb_walk_make_unmake(p, depth - 1)) return false;
        unmakeMove(p, ml.m[i], u);
        if (!bb_position_core_equal(p, before)) return false;
    }
    return true;
}

static bool bb_magic_exhaustive_check() {
    if (!bb_init_magic_tables()) return false;
    for (int kind = 0; kind < 2; ++kind) {
        const bool bishop = kind == 0;
        const BBMagicEntry* info = bishop
            ? bb_bishop_magic_info : bb_rook_magic_info;
        for (int sq64 = 0; sq64 < 64; ++sq64) {
            const Bitboard mask = info[sq64].mask;
            const uint32_t variations = 1u << bb_popcount64(mask);
            for (uint32_t i = 0; i < variations; ++i) {
                const Bitboard occupied =
                    bb_magic_subset_from_index(i, mask);
                const Bitboard expected =
                    bb_sliding_attacks_reference(sq64, occupied, bishop);
                const Bitboard actual = bb_magic_lookup(info[sq64], occupied);
                if (expected != actual) return false;
            }
        }
    }
    return true;
}

static void run_bitboard_tests() {
    int pass = 0, fail = 0;
    auto result = [&](bool ok, const char* name) {
        if (ok) { ++pass; std::printf("info string BB %s OK\n", name); }
        else { ++fail; std::printf("info string BB %s FAIL\n", name); }
    };

    bool mapping = true;
    for (int sq64 = 0; sq64 < 64; ++sq64) {
        int sq88 = sq88_from_bb_sq(sq64);
        if (!on_board(sq88) || bb_sq_from_sq88(sq88) != sq64) mapping = false;
    }
    result(mapping, "square_mapping");

    bb_init_attack_tables();
    result(bb_magic_exhaustive_check(), "magic_sliding_tables");
    bool attack_tables = true;
    for (int sq64 = 0; sq64 < 64; ++sq64) {
        if (bb_pawn_attack_mask(WHITE, sq64) !=
            bb_reference_pawn_attacks(WHITE, sq64) ||
            bb_pawn_attack_mask(BLACK, sq64) !=
            bb_reference_pawn_attacks(BLACK, sq64) ||
            bb_knight_attack_mask(sq64) !=
            bb_reference_step_attacks(sq64, KNIGHT_DIRS, 8) ||
            bb_king_attack_mask(sq64) !=
            bb_reference_step_attacks(sq64, KING_DIRS, 8)) {
            attack_tables = false;
            break;
        }
    }
    // Differentially compare every non-sliding table entry with the existing
    // 0x88 isSquareAttacked() implementation using isolated single-attacker
    // positions. This is deliberately independent of the table constructor.
    for (int color = WHITE; color <= BLACK && attack_tables; ++color) {
        for (int type : { PAWN, KNIGHT, KING }) {
            for (int from64 = 0; from64 < 64 && attack_tables; ++from64) {
                for (int target64 = 0; target64 < 64; ++target64) {
                    Position q;
                    clear_pos(q);
                    bb_piece_ref(q, sq88_from_bb_sq(from64)) = make_pc(type, color);
                    const int target88 = sq88_from_bb_sq(target64);
                    const bool old_result = isSquareAttacked_0x88(q, target88, color);
                    const Bitboard target = bb_square(target64);
                    const Bitboard mask = type == PAWN
                        ? bb_pawn_attack_mask(color, from64)
                        : type == KNIGHT
                            ? bb_knight_attack_mask(from64)
                            : bb_king_attack_mask(from64);
                    if (old_result != ((mask & target) != 0)) {
                        attack_tables = false;
                        break;
                    }
                }
            }
        }
    }
    result(attack_tables, "non_sliding_attack_tables");

    // Differential sliding-attack test. For every source square and many
    // deterministic random occupancies, build an isolated 0x88 position with
    // one white slider and compare every target square against the bitboard
    // ray implementation. Black pawns are blockers only; they are never
    // counted as white attackers by isSquareAttacked().
    bool sliding_attacks = true;
    uint64_t random_state = 0x6a09e667f3bcc909ULL;
    for (int kind = 0; kind < 3 && sliding_attacks; ++kind) {
        const int piece = kind == 0 ? BISHOP : kind == 1 ? ROOK : QUEEN;
        for (int from64 = 0; from64 < 64 && sliding_attacks; ++from64) {
            for (int sample = 0; sample < 96 && sliding_attacks; ++sample) {
                Bitboard occupied = splitmix64(random_state);
                occupied &= ~bb_square(from64);
                Position q;
                clear_pos(q);
                const int from88 = sq88_from_bb_sq(from64);
                bb_piece_ref(q, from88) = make_pc(piece, WHITE);
                for (int sq64 = 0; sq64 < 64; ++sq64) {
                    if (occupied & bb_square(sq64))
                        bb_piece_ref(q, sq88_from_bb_sq(sq64)) = make_pc(PAWN, BLACK);
                }
                const Bitboard mask = kind == 0
                    ? bb_bishop_attack_mask(from64, occupied)
                    : kind == 1
                        ? bb_rook_attack_mask(from64, occupied)
                        : bb_queen_attack_mask(from64, occupied);
                for (int target64 = 0; target64 < 64; ++target64) {
                    const bool old_result = isSquareAttacked_0x88(
                        q, sq88_from_bb_sq(target64), WHITE);
                    const bool bb_result = (mask & bb_square(target64)) != 0;
                    if (old_result != bb_result) {
                        sliding_attacks = false;
                        break;
                    }
                }
            }
        }
    }
    result(sliding_attacks, "sliding_attack_reference");

    bool attacked_square = true;
    int attacked_compared = 0;
    const char* attack_fens[] = {
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/8/3p4/2b1r3/3Q4/2N5/4K3/8 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    };
    for (const char* fen : attack_fens) {
        Position q;
        if (!parse_fen(q, fen)) { attacked_square = false; break; }
        for (int color = WHITE; color <= BLACK && attacked_square; ++color) {
            for (int target64 = 0; target64 < 64; ++target64) {
                const int target88 = sq88_from_bb_sq(target64);
                const bool old_result = isSquareAttacked_0x88(q, target88, color);
                const bool new_result = isSquareAttacked(q, target88, color);
                ++attacked_compared;
                if (old_result != new_result) {
                    attacked_square = false;
                    break;
                }
            }
        }
        if (!attacked_square) break;
    }
    result(attacked_square && attacked_compared > 0, "attacked_square_differential");

    // Stage 7 differential test: compare the complete pseudo-legal move set
    // from the old 0x88 generator with the active bitboard generator. Flags
    // are part of the key, so captures, double pushes, promotions, castling,
    // and en passant cannot silently compare equal as plain from/to moves.
    bool pseudo_differential = true;
    int pseudo_positions = 0;
    const char* pseudo_fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/P6p/1p6/3pPp2/8/8/p6P/4K2k w - f6 0 1",
        "7k/4P3/8/8/8/8/4p3/K7 b - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1",
    };
    for (const char* fen : pseudo_fens) {
        Position q;
        if (!parse_fen(q, fen)) { pseudo_differential = false; break; }
        MoveList old_list, new_list;
        generateMoves_0x88(q, old_list);
        generateMoves(q, new_list);
        ++pseudo_positions;
        if (!bb_move_lists_equal(old_list, new_list)) {
            pseudo_differential = false;
            break;
        }
    }
    // Continue through a deterministic legal walk to cover positions that are
    // not represented by the fixed FENs above.
    if (pseudo_differential) {
        Position q;
        set_start(q);
        uint64_t walk_state = 0xbb67ae8584caa73bULL;
        for (int ply = 0; ply < 120 && pseudo_differential; ++ply) {
            MoveList old_list, new_list;
            generateMoves_0x88(q, old_list);
            generateMoves(q, new_list);
            ++pseudo_positions;
            if (!bb_move_lists_equal(old_list, new_list)) {
                pseudo_differential = false;
                break;
            }
            MoveList legal;
            generateLegalMoves(q, legal);
            if (legal.count == 0) break;
            const int pick = int(splitmix64(walk_state) % (uint64_t)legal.count);
            Undo u;
            makeMove(q, legal.m[pick], u);
        }
    }
    result(pseudo_differential, "pseudo_move_differential");
    std::printf("info string BB pseudo_positions_compared %d\n", pseudo_positions);

    // Stage 8 differential test: active legal filtering must match the old
    // 0x88 legal path, including pins, checks, castling transit squares,
    // en-passant discovered checks, and promotions.
    bool legal_differential = true;
    int legal_positions = 0;
    const char* legal_fens[] = {
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/8/8/r2pP2K/8/8/8/4k3 w - d6 0 1",
        "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1",
        "7k/P7/8/8/8/8/8/4K3 w - - 0 1",
        "4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1",
        "4r1k1/8/8/8/8/8/4K3/8 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    };
    for (const char* fen : legal_fens) {
        Position q;
        if (!parse_fen(q, fen)) { legal_differential = false; break; }
        MoveList old_list, new_list;
        generateLegalMoves_0x88(q, old_list);
        generateLegalMoves(q, new_list);
        ++legal_positions;
        if (!bb_move_lists_equal(old_list, new_list)) {
            legal_differential = false;
            break;
        }
    }
    if (legal_differential) {
        Position q;
        set_start(q);
        uint64_t legal_walk_state = 0x3c6ef372fe94f82bULL;
        for (int ply = 0; ply < 120 && legal_differential; ++ply) {
            MoveList old_list, new_list;
            generateLegalMoves_0x88(q, old_list);
            generateLegalMoves(q, new_list);
            ++legal_positions;
            if (!bb_move_lists_equal(old_list, new_list)) {
                legal_differential = false;
                break;
            }
            if (new_list.count == 0) break;
            const int pick = int(splitmix64(legal_walk_state) %
                (uint64_t)new_list.count);
            Undo u;
            makeMove(q, new_list.m[pick], u);
        }
    }
    result(legal_differential, "legal_move_differential");
    std::printf("info string BB legal_positions_compared %d\n", legal_positions);

    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/P6p/1p6/3pPp2/8/8/p6P/4K2k w - f6 0 1",
        "7k/4P3/8/8/8/8/4p3/K7 b - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    };
    bool rebuilt = true;
    for (const char* fen : fens) {
        Position p;
        if (!parse_fen(p, fen) || !bb_state_matches(p)) rebuilt = false;
    }
    result(rebuilt, "fen_rebuild");

    bool incremental = true;
    for (const char* fen : fens) {
        Position p;
        if (!parse_fen(p, fen) || !bb_walk_make_unmake(p, 2)) {
            incremental = false;
            break;
        }
    }
    result(incremental, "incremental_make_unmake");
    std::printf("info string BB TESTS DONE pass %d fail %d\n", pass, fail);
    std::fflush(stdout);
}

static int gen_fen(const char* fen, MoveList& ml) {
    Position p;
    if (!parse_fen(p, fen)) { ml.clear(); return -1; }
    generateMoves(p, ml);
    return ml.count;
}
static int count_from(const char* fen, const char* sq) {
    MoveList ml; gen_fen(fen, ml);
    int s = sq_of(sq), n = 0;
    for (int i = 0; i < ml.count; ++i) if (ml.m[i].from == s) ++n;
    return n;
}
static int find_move(const char* fen, const char* mv) {
    MoveList ml; gen_fen(fen, ml); char b[8];
    for (int i = 0; i < ml.count; ++i) { move_str(ml.m[i], b); if (!std::strcmp(b, mv)) return 0x100 | ml.m[i].flags; }
    return 0;
}
static bool has(const char* fen, const char* mv) { return find_move(fen, mv) != 0; }
static int total(const char* fen) { MoveList ml; return gen_fen(fen, ml); }
static bool peq(const Position& a, const Position& b) {
    return !std::memcmp(a.piece, b.piece, sizeof a.piece) && a.sideToMove == b.sideToMove
        && a.castle == b.castle && a.ep == b.ep && a.halfmove == b.halfmove && a.fullmove == b.fullmove
        && a.ksq[WHITE] == b.ksq[WHITE] && a.ksq[BLACK] == b.ksq[BLACK];
}
static bool getmv(const Position& p, const char* s, Move& m) {
    MoveList ml; generateMoves(p, ml); char b[8];
    for (int i = 0; i < ml.count; ++i) { move_str(ml.m[i], b); if (!std::strcmp(b, s)) { m = ml.m[i]; return true; } }
    return false;
}
static void rt(const char* fen, const char* s, const char* n) {
    Position p, o; Move m{}; Undo u;
    bool ok = parse_fen(p, fen) && getmv(p, s, m);
    check(ok, n);
    if (!ok) return;
    o = p; makeMove(p, m, u); unmakeMove(p, m, u);
    check(peq(p, o), n);
}
static void test_mu() {
    Position p, o; Move m, m1, m2; Undo u, u1, u2;
    set_start(p); o = p; getmv(p, "e2e4", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("e4")) == make_pc(PAWN, WHITE) && bb_piece_ref(p, sq_of("e2")) == EMPTY, "e2e4 pcs");
    check(p.ep == sq_of("e3") && p.sideToMove == BLACK && p.halfmove == 0 && p.fullmove == 1, "e2e4 st");
    unmakeMove(p, m, u); check(peq(p, o), "e2e4 rt");
    parse_fen(p, "8/8/8/4p3/3P4/8/8/8 w - - 5 3"); o = p;
    getmv(p, "d4e5", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("e5")) == make_pc(PAWN, WHITE) && bb_piece_ref(p, sq_of("d4")) == EMPTY && p.halfmove == 0, "cap");
    unmakeMove(p, m, u); check(peq(p, o), "cap rt");
    parse_fen(p, "8/4P3/8/8/8/8/8/8 w - - 0 1"); o = p;
    getmv(p, "e7e8q", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("e8")) == make_pc(QUEEN, WHITE) && bb_piece_ref(p, sq_of("e7")) == EMPTY, "promo");
    unmakeMove(p, m, u); check(peq(p, o), "promo rt");
    parse_fen(p, "5r2/4P3/8/8/8/8/8/8 w - - 0 1"); o = p;
    getmv(p, "e7f8q", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("f8")) == make_pc(QUEEN, WHITE), "cpromo");
    unmakeMove(p, m, u); check(peq(p, o) && bb_piece_ref(p, sq_of("f8")) == make_pc(ROOK, BLACK), "cpromo rt");
    parse_fen(p, "8/8/8/3pP3/8/8/8/8 w - d6 0 1"); o = p;
    getmv(p, "e5d6", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("d6")) == make_pc(PAWN, WHITE) && bb_piece_ref(p, sq_of("d5")) == EMPTY
        && bb_piece_ref(p, sq_of("e5")) == EMPTY && p.ep == NO_SQ, "ep");
    unmakeMove(p, m, u); check(peq(p, o) && bb_piece_ref(p, sq_of("d5")) == make_pc(PAWN, BLACK), "ep rt");
    parse_fen(p, "rnbqkbnr/ppp1p1pp/8/8/3pPp2/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"); o = p;
    getmv(p, "d4e3", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("e3")) == make_pc(PAWN, BLACK) && bb_piece_ref(p, sq_of("e4")) == EMPTY
        && bb_piece_ref(p, sq_of("d4")) == EMPTY, "epb");
    unmakeMove(p, m, u); check(peq(p, o), "epb rt");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"); o = p;
    getmv(p, "e1g1", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("g1")) == make_pc(KING, WHITE) && bb_piece_ref(p, sq_of("f1")) == make_pc(ROOK, WHITE)
        && bb_piece_ref(p, sq_of("e1")) == EMPTY && bb_piece_ref(p, sq_of("h1")) == EMPTY && !(p.castle & (WK | WQ)), "WOO");
    unmakeMove(p, m, u); check(peq(p, o), "WOO rt");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"); o = p;
    getmv(p, "e1c1", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("c1")) == make_pc(KING, WHITE) && bb_piece_ref(p, sq_of("d1")) == make_pc(ROOK, WHITE)
        && bb_piece_ref(p, sq_of("a1")) == EMPTY && bb_piece_ref(p, sq_of("e1")) == EMPTY, "WOOO");
    unmakeMove(p, m, u); check(peq(p, o), "WOOO rt");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"); o = p;
    getmv(p, "e8g8", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("g8")) == make_pc(KING, BLACK) && bb_piece_ref(p, sq_of("f8")) == make_pc(ROOK, BLACK)
        && bb_piece_ref(p, sq_of("e8")) == EMPTY && bb_piece_ref(p, sq_of("h8")) == EMPTY, "BOO");
    unmakeMove(p, m, u); check(peq(p, o), "BOO rt");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"); o = p;
    getmv(p, "e8c8", m); makeMove(p, m, u);
    check(bb_piece_ref(p, sq_of("c8")) == make_pc(KING, BLACK) && bb_piece_ref(p, sq_of("d8")) == make_pc(ROOK, BLACK)
        && bb_piece_ref(p, sq_of("a8")) == EMPTY && bb_piece_ref(p, sq_of("e8")) == EMPTY, "BOOO");
    unmakeMove(p, m, u); check(peq(p, o), "BOOO rt");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    getmv(p, "e1e2", m); makeMove(p, m, u);
    check(!(p.castle & (WK | WQ)) && (p.castle & (BK | BQ)) == (BK | BQ), "king rights");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    getmv(p, "a1a2", m); makeMove(p, m, u);
    check(!(p.castle & WQ) && (p.castle & WK), "rook a1");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    getmv(p, "h1h2", m); makeMove(p, m, u);
    check(!(p.castle & WK) && (p.castle & WQ), "rook h1");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
    getmv(p, "a8a7", m); makeMove(p, m, u);
    check(!(p.castle & BQ) && (p.castle & BK), "rook a8");
    parse_fen(p, "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
    getmv(p, "h8h7", m); makeMove(p, m, u);
    check(!(p.castle & BK) && (p.castle & BQ), "rook h8");
    parse_fen(p, "8/8/8/8/8/8/r7/R3K3 b Q - 0 1");
    getmv(p, "a2a1", m); makeMove(p, m, u);
    check(p.castle == 0, "cap a1");
    parse_fen(p, "8/8/8/8/8/8/7r/4K2R b K - 0 1");
    getmv(p, "h2h1", m); makeMove(p, m, u);
    check(p.castle == 0, "cap h1");
    parse_fen(p, "r3k3/8/8/8/8/8/8/R7 w q - 0 1");
    getmv(p, "a1a8", m); makeMove(p, m, u);
    check(p.castle == 0, "cap a8");
    set_start(p); getmv(p, "g1f3", m); makeMove(p, m, u);
    check(p.halfmove == 1 && p.fullmove == 1 && p.sideToMove == BLACK && p.ep == NO_SQ, "nf3 clk");
    getmv(p, "b8c6", m); makeMove(p, m, u2);
    check(p.halfmove == 2 && p.fullmove == 2 && p.sideToMove == WHITE, "nc6 clk");
    unmakeMove(p, m, u2);
    check(p.halfmove == 1 && p.fullmove == 1 && p.sideToMove == BLACK, "clk unmake");
    set_start(p); o = p;
    getmv(p, "e2e4", m1); makeMove(p, m1, u1);
    getmv(p, "e7e5", m2); makeMove(p, m2, u2);
    unmakeMove(p, m2, u2); unmakeMove(p, m1, u1);
    check(peq(p, o), "2ply rt");
    rt("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "b1c3", "rt Nc3");
    rt("rnbqkbnr/ppp1p1pp/8/8/3pPp2/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "f4e3", "rt epb2");
    rt("8/8/8/8/8/8/4p3/8 b - - 0 1", "e2e1q", "rt bpr");
    rt("5r2/4P3/8/8/8/8/8/8 w - - 0 1", "e7f8n", "rt cpn");
    rt("8/8/8/8/8/8/4p3/5R2 b - - 0 1", "e2f1q", "rt bcp");
    rt("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", "rt WOO2");
    rt("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", "rt BOOO2");
    rt("8/8/8/3pP3/8/8/8/8 w - d6 0 1", "e5d6", "rt ep2");
}
static int legal_total(const char* fen) {
    Position p; if (!parse_fen(p, fen)) return -1;
    MoveList ml; generateLegalMoves(p, ml); return ml.count;
}
static bool has_legal(const char* fen, const char* mv) {
    Position p; if (!parse_fen(p, fen)) return false;
    MoveList ml; generateLegalMoves(p, ml); char b[8];
    for (int i = 0; i < ml.count; ++i) { move_str(ml.m[i], b); if (!std::strcmp(b, mv)) return true; }
    return false;
}
static void test_legal() {
    Position p;
    parse_fen(p, "4k3/8/8/8/8/8/8/4R2K b - - 0 1");
    check(inCheck(p, BLACK) && !inCheck(p, WHITE), "check rook");
    parse_fen(p, "4k3/8/8/1B6/8/8/8/4K3 b - - 0 1");
    check(inCheck(p, BLACK), "check bishop");
    parse_fen(p, "4k3/3N4/8/8/8/8/8/4K3 b - - 0 1");
    check(inCheck(p, BLACK), "check knight");
    parse_fen(p, "4k3/3P4/8/8/8/8/8/4K3 b - - 0 1");
    check(inCheck(p, BLACK), "check pawn w");
    parse_fen(p, "4k3/8/8/8/8/8/3p4/4K3 w - - 0 1");
    check(inCheck(p, WHITE), "check pawn b");
    check(legal_total("8/8/8/8/8/8/r7/4K3 w - - 0 1") == 2, "king attacked sq filtered");
    check(!has_legal("4r3/8/8/8/8/8/4N3/4K3 w - - 0 1", "e2d4"), "pinned knight cannot move");
    check(has_legal("7b/8/8/8/8/2B5/8/K7 w - - 0 1", "c3h8"), "pinned bishop capture legal");
    check(!has_legal("7b/8/8/8/8/2B5/8/K7 w - - 0 1", "c3d4"), "pinned bishop off-diagonal");
    check(has_legal("4r3/8/8/8/8/8/8/1B2K3 w - - 0 1", "b1e4"), "block check legal");
    check(!has_legal("4r3/8/8/8/8/8/8/1B2K3 w - - 0 1", "b1a2"), "non-blocking illegal");
    check(!has_legal("8/8/8/r2pP2K/8/8/8/8 w - d6 0 1", "e5d6"), "ep pin illegal");
    check(has_legal("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1", "e5d6"), "ep legal");
    const char* C_CHK = "r3k2r/8/8/8/8/8/4r3/R3K2R w KQkq - 0 1";
    check(!has_legal(C_CHK, "e1g1") && !has_legal(C_CHK, "e1c1"), "castle out of check illegal");
    const char* C_TRN = "r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1";
    check(!has_legal(C_TRN, "e1g1") && has_legal(C_TRN, "e1c1"), "castle through check illegal");
    const char* C_DST = "r3k2r/8/8/8/8/6r1/8/R3K2R w KQkq - 0 1";
    check(!has_legal(C_DST, "e1g1"), "castle into check illegal");
    check(legal_total("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") == 20, "startpos 20 legal moves");
    check(legal_total("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1") == 20, "startpos black 20 legal moves");
}
static void zrt(const char* fen, const char* mv, const char* name) {
    Position p; Move m{}; Undo u;
    bool ok = parse_fen(p, fen) && getmv(p, mv, m);
    check(ok, name);
    if (!ok) return;
    uint64_t h = p.key;
    makeMove(p, m, u);
    check(p.key == computeHash(p), name);
    unmakeMove(p, m, u);
    check(p.key == h, name);
}
static void test_z() {
    Position a, b;
    parse_fen(a, "8/8/8/8/8/8/8/4K3 w - - 0 1");
    parse_fen(b, "8/8/8/8/8/8/8/4K3 w - - 0 1");
    check(a.key == b.key && a.key == computeHash(a), "z same");
    parse_fen(b, "8/8/8/8/8/8/8/4K3 b - - 0 1");
    check(a.key != b.key, "z side");
    parse_fen(b, "8/8/8/8/8/8/8/3K4 w - - 0 1");
    check(a.key != b.key, "z pieces");
    parse_fen(a, "8/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    parse_fen(b, "8/8/8/8/8/8/8/R3K2R w K - 0 1");
    check(a.key != b.key, "z castle");
    // build0.3: a raw ep square with no actual capturing pawn on the board
    // must NOT change the hash. This bare-king position has no pawns at all,
    // so "e3" is a dead ep square and both FENs must hash identically.
    parse_fen(a, "8/8/8/8/8/8/8/4K3 w - - 0 1");
    parse_fen(b, "8/8/8/8/8/8/8/4K3 w - e3 0 1");
    check(a.key == b.key, "z ep dead square canonicalized");
    zrt("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", "z normal");
    zrt("8/8/8/4p3/3P4/8/8/8 w - - 5 3", "d4e5", "z capture");
    zrt("8/4P3/8/8/8/8/8/8 w - - 0 1", "e7e8q", "z promotion");
    zrt("5r2/4P3/8/8/8/8/8/8 w - - 0 1", "e7f8q", "z cap-promo");
    zrt("8/8/8/3pP3/8/8/8/8 w - d6 0 1", "e5d6", "z en-passant");
    zrt("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", "z castling");
}
// ============================================================================
// FROZEN SEARCH A
// ============================================================================
constexpr int SEARCH_INFINITE_SCORE = 30000;
constexpr int SEARCH_MATE_SCORE = 29000;
static const int search_piece_vals[] = { 0, 100, 320, 330, 500, 900, 20000 };
// build0.9: pruning thresholds. FUTILITY_MARGIN indexed by remaining depth
// (index 0 unused). DELTA_NPM_MIN: delta pruning is disabled when the total
// non-pawn material of BOTH sides is <= this value (late endgames are about
// deep resources, not material counting). Margins assume Eval A error
// profile (mine iter-1: MAE 128cp, P90 263cp) — revisit after eval changes.
static const int SEARCH_FUTILITY_MARGIN[3] = { 0, 200, 450 };
static const int SEARCH_DELTA_MARGIN = 200;
static const int SEARCH_DELTA_NPM_MIN = 1300;
// ============================================================================
// FROZEN EVALUATION A
// ============================================================================
static const int EV_MG[7] = { 0, 82, 337, 365, 477, 1025, 0 };
static const int EV_EG[7] = { 0, 94, 281, 297, 512, 936, 0 };
static const int EV_PHASE[7] = { 0, 0, 1, 1, 2, 4, 0 };
static inline int ev_sign(int c) {
    return c == WHITE ? 1 : -1;
}
static inline int ev_relrank(int sq, int c) {
    int r = rank_of(sq);
    return c == WHITE ? r : 7 - r;
}
static inline int ev_center(int sq) {
    int f = file_of(sq), r = rank_of(sq);
    int df = f < 4 ? 3 - f : f - 4;
    int dr = r < 4 ? 3 - r : r - 4;
    return 6 - df - dr;
}
static inline int ev_blend(int mg, int eg, int phase) {
    return (mg * phase + eg * (24 - phase)) / 24;
}
static void ev_mark_attack(bool a[128], int sq) {
    if (on_board(sq)) a[sq] = true;
}
static int ev_slider_attacks(const Position& p, int sq, const int* d, int n,
    bool a[128], int us) {
    int mob = 0;
    for (int i = 0; i < n; ++i) {
        for (int to = sq + d[i]; on_board(to); to += d[i]) {
            int pc = bb_piece_ref(p, to);
            ev_mark_attack(a, to);
            if (!pc) {
                ++mob;
                continue;
            }
            if (color_of(pc) != us) ++mob;
            break;
        }
    }
    return mob;
}
static int ev_step_attacks(const Position& p, int sq, const int* d, int n,
    bool a[128], int us) {
    int mob = 0;
    for (int i = 0; i < n; ++i) {
        int to = sq + d[i];
        if (!on_board(to)) continue;
        a[to] = true;
        if (!bb_piece_ref(p, to) || color_of(bb_piece_ref(p, to)) != us) ++mob;
    }
    return mob;
}
// build0.5: passed-pawn geometry uses absolute board ranks. Only enemy
// pawns on the same or adjacent files strictly ahead toward promotion block.
static bool ev_is_passed_pawn(const Position& p, int c, int sq) {
    const int f = file_of(sq);
    const int enemy = c ^ 1;
    const int step = (c == WHITE) ? 1 : -1;

    for (int ef = std::max(0, f - 1);
        ef <= std::min(7, f + 1);
        ++ef) {
        for (int rr = rank_of(sq) + step;
            valid_rank(rr);
            rr += step) {
            if (bb_piece_ref(p, sq88(ef, rr)) == make_pc(PAWN, enemy))
                return false;
        }
    }
    return true;
}

static int ev_material_scale(const int count[2][7]) {
    int pawns = count[WHITE][PAWN] + count[BLACK][PAWN];
    int rq = count[WHITE][ROOK] + count[BLACK][ROOK] +
        count[WHITE][QUEEN] + count[BLACK][QUEEN];
    int minors = count[WHITE][KNIGHT] + count[BLACK][KNIGHT] +
        count[WHITE][BISHOP] + count[BLACK][BISHOP];
    if (pawns == 0 && rq == 0 && minors <= 1) return 0;
    if (pawns == 0 && rq == 0 && minors == 2 &&
        count[WHITE][BISHOP] == 1 && count[BLACK][BISHOP] == 1) return 0;
    return 128;
}
static const short W7_PST_MG[6][64] = {
    {   82,   84,   86,   95,   95,   86,   84,   82,
        95,  103,  109,  103,  111,  144,  155,  103,
        92,   81,  117,  101,  116,  112,  127,  100,
        88,   89,  113,  133,  133,  128,  106,   75,
       112,  108,  115,  128,  140,  132,  116,   93,
       113,  116,  144,  129,  156,  207,  116,  111,
       163,  114,   99,  133,  152,  161,   72,   51,
       135,  137,  139,  155,  155,  139,  137,  135,},
    {  403,  450,  425,  451,  451,  459,  446,  387,
       451,  435,  469,  486,  485,  482,  472,  467,
       452,  472,  494,  490,  503,  502,  505,  458,
       446,  443,  477,  461,  483,  481,  464,  436,
       468,  473,  462,  505,  458,  490,  440,  479,
       441,  463,  468,  508,  533,  562,  495,  423,
       366,  415,  540,  475,  531,  530,  447,  441,
       179,  393,  392,  397,  485,  320,  301,  294,},
    {  453,  428,  459,  453,  439,  433,  438,  455,
       419,  484,  455,  455,  462,  476,  501,  451,
       473,  471,  472,  444,  460,  483,  464,  477,
       437,  438,  441,  460,  465,  442,  451,  424,
       413,  431,  435,  448,  448,  414,  441,  432,
       417,  434,  486,  436,  500,  461,  477,  454,
       393,  436,  406,  412,  424,  486,  408,  450,
       402,  366,  378,  342,  300,  399,  409,  346,},
    {  624,  625,  634,  636,  644,  648,  612,  641,
       600,  606,  601,  615,  619,  631,  660,  577,
       592,  598,  607,  620,  620,  634,  663,  636,
       589,  580,  570,  580,  597,  601,  632,  609,
       575,  600,  606,  617,  595,  626,  644,  646,
       602,  596,  583,  609,  634,  659,  706,  663,
       601,  578,  616,  666,  624,  682,  710,  676,
       643,  681,  657,  648,  675,  729,  707,  632,},
    { 1164, 1153, 1165, 1175, 1164, 1145, 1116, 1181,
      1156, 1163, 1164, 1168, 1175, 1189, 1189, 1163,
      1143, 1168, 1143, 1147, 1147, 1168, 1174, 1162,
      1133, 1128, 1124, 1110, 1134, 1134, 1154, 1137,
      1113, 1117, 1108, 1090, 1098, 1109, 1110, 1148,
      1113, 1106, 1133, 1114, 1146, 1115, 1215, 1127,
      1107, 1096, 1073, 1037, 1041, 1167, 1114, 1266,
      1122, 1114, 1147, 1178, 1199, 1178, 1081, 1122,},
    {  -48,   27,   41,  -41,  -23,   -8,   34,   34,
        13,  -55,  -67, -115,  -97,  -82,  -13,  -15,
        13,   32,  -73, -108,  -96,  -51,  -21,  -63,
        -2,   48,   -2,  -94,  -75,  -78,  -30, -110,
         0,   15,   42,  -19,  -33,   11,   17,  -63,
        70,  187,   48,   73,    3,   79,  116,  -21,
        38,  164,  110,   68,   63,  153,   -4,  -33,
       157,   62,  173,  116,  127,  135,   69,   55,},
};
static const short W7_PST_EG[6][64] = {
    {   94,   95,   96,   97,   97,   96,   95,   94,
       112,  108,  109,  108,  108,  104,   91,   89,
       100,  103,   93,  100,  101,  100,   90,   86,
       112,  111,   92,   83,   89,   89,   99,   96,
       126,  120,  104,   86,   90,   97,  111,  108,
       150,  137,  119,   85,   75,   84,  115,  122,
       211,  212,  195,  153,  142,  148,  203,  200,
       182,  183,  184,  192,  192,  184,  183,  182,},
    {  274,  302,  329,  329,  320,  330,  295,  304,
       304,  321,  321,  330,  328,  329,  314,  298,
       312,  334,  332,  358,  355,  336,  322,  303,
       332,  348,  368,  377,  379,  358,  340,  334,
       334,  356,  376,  388,  394,  383,  367,  331,
       311,  330,  373,  369,  346,  346,  328,  336,
       316,  334,  317,  342,  317,  311,  317,  297,
       320,  283,  318,  336,  317,  325,  315,  249,},
    {  289,  308,  292,  306,  311,  313,  295,  292,
       314,  289,  302,  308,  312,  300,  293,  289,
       293,  317,  311,  327,  320,  308,  302,  294,
       305,  314,  329,  316,  318,  323,  304,  308,
       322,  326,  327,  332,  327,  326,  310,  315,
       324,  313,  309,  334,  311,  330,  318,  313,
       310,  309,  318,  309,  321,  304,  320,  284,
       308,  303,  309,  323,  330,  321,  308,  332,},
    {  597,  596,  594,  595,  585,  586,  593,  563,
       593,  596,  606,  600,  598,  593,  585,  598,
       593,  596,  600,  597,  597,  589,  578,  573,
       605,  611,  619,  620,  614,  609,  593,  593,
       620,  609,  616,  610,  614,  610,  601,  602,
       621,  623,  624,  618,  609,  604,  595,  599,
       624,  642,  630,  613,  617,  605,  594,  599,
       618,  603,  614,  611,  604,  589,  593,  611,},
    {  964,  968,  948,  945,  971,  959,  958,  924,
       979,  984,  972,  978,  970,  928,  926,  963,
       951,  940, 1008,  984,  995,  994, 1007,  981,
       989, 1004, 1009, 1043, 1026, 1012, 1009, 1017,
       984, 1012, 1017, 1061, 1076, 1076, 1057, 1025,
       973,  996, 1018, 1033, 1070, 1052,  988, 1023,
      1014, 1004, 1047, 1107, 1091, 1074, 1063,  920,
       986, 1009, 1004,  999,  982, 1012, 1020, 1030,},
    {    8,   13,   20,   31,   36,   32,    9,  -22,
        -2,   36,   52,   66,   66,   63,   39,   18,
        -9,   19,   53,   69,   69,   59,   41,   32,
        -9,   18,   44,   67,   69,   62,   43,   36,
         4,   32,   40,   57,   56,   55,   51,   45,
         5,   22,   41,   37,   53,   57,   56,   50,
         5,   19,   19,   27,   31,   38,   60,   45,
       -66,    1,  -21,    4,    5,   12,   14,  -12,},
};
static const short W7_MG[100] = {
       -1,  -20,    0,    4,    4,   -5,   10,    3,   46,    0,
        4,   -3,   12,  -15,   58,   47,   16,   11,    9,    4,
       36,   26,   12,  -31, -102, -194, -208, -331, -168,   25,
       19,   48,  -31,   66,   -6,  -25,   -5,    1,    9,    9,
       14,   18,   20,  -41,  -14,   -6,   -1,    3,    3,   11,
       13,   18,   24,   33,   36,   42,   55,  -29,   -5,   -1,
        1,    5,   11,   13,   12,   14,   21,   22,   28,   24,
       38,   56,  -37,    5,   10,    9,   12,   18,   21,   20,
       17,   23,   28,   26,   28,   30,   31,   40,   38,   36,
       47,   47,   34,   48,   56,   75,   39,   72,  269, -231,

};
static const short W7_EG[100] = {
       -9,    0,    0,   26,   14,   21,   33,   92,   75,    0,
      -49,   21,   13,   -5,   57,  -10,   10,   -3,    0,   -2,
      -37,  -10,   -4,    3,   40,   69,   79,  106,   14,   -8,
       -2,    3,    8,  -47,   21,    7,    0,    1,    0,    6,
        4,    2,   -4,  -13,  -23,  -24,  -13,   -4,   10,   12,
       23,   24,   25,   27,   29,   45,   55,   10,  -20,  -13,
       -2,    1,    1,    3,    8,   14,   13,   19,   24,   35,
       31,   30, -165,   21,   17,   15,   21,   47,   45,   54,
       79,   88,   94,  107,  110,  120,  124,  112,  123,  139,
      133,  135,  154,  142,  132,  123,  129,  129,   32,  269,

};

enum {
    E7_DOUBLED = 384, E7_ISOLATED = 385, E7_PASSED = 386, E7_PASS_FREE = 394,
    E7_PASS_FREE_R = 395, E7_PAWN_DEF = 396, E7_PAWN_BLOCKED = 397, E7_BISHOP_PAIR = 398,
    E7_ROOK_OPEN = 399, E7_ROOK_SEMI = 400, E7_KNIGHT_OUT = 401, E7_PIECE_DEF = 402,
    E7_CENTER = 403, E7_KRING = 404, E7_KSHIELD1 = 413, E7_KSHIELD2 = 414,
    E7_KOFFHOME = 415, E7_KCENTRALHOME = 416, E7_CASTLE = 417,
    E7_MOB_N = 418, E7_MOB_B = 427, E7_MOB_R = 441, E7_MOB_Q = 456
};
static const int E7_MOB_MAX[7] = { 0, 0, 8, 13, 14, 27, 0 };
static const int E7_MOB_BASE[7] = { 0, 0, E7_MOB_N, E7_MOB_B, E7_MOB_R, E7_MOB_Q, 0 };
static inline int e7_sq64(int sq, int c) {
    int f = file_of(sq), r = rank_of(sq);
    if (c == BLACK) r = 7 - r;
    return r * 8 + f;
}
int evaluate(const Position& pos) {
    bool attacked[2][128] = {};
    bool pawnAttack[2][128] = {};
    int pawns[2][8] = {};
    int count[2][7] = {};
    int kingSq[2] = { NO_SQ, NO_SQ };
    int mobility[2][7] = {};
    int phase = 0, mg = 0, eg = 0;
#define A7F(i, n) do { mg += sg * (n) * W7_MG[(i) - 384]; \
                       eg += sg * (n) * W7_EG[(i) - 384]; } while (0)
    for (int r = 0; r < 8; ++r)
        for (int f = 0; f < 8; ++f) {
            int sq = sq88(f, r), pc = bb_piece_ref(pos, sq);
            if (!pc) continue;
            int c = color_of(pc), pt = piece_of(pc);
            ++count[c][pt];
            phase += EV_PHASE[pt];
            if (pt == PAWN) {
                ++pawns[c][f];
                int dir = c == WHITE ? 16 : -16;
                ev_mark_attack(pawnAttack[c], sq + dir - 1);
                ev_mark_attack(pawnAttack[c], sq + dir + 1);
            }
            if (pt == KING) kingSq[c] = sq;
        }
    if (phase > 24) phase = 24;
    for (int c = WHITE; c <= BLACK; ++c)
        for (int sq = 0; sq < 128; ++sq) {
            if (!on_board(sq)) { sq += 7; continue; }
            int pc = bb_piece_ref(pos, sq);
            if (!pc || color_of(pc) != c) continue;
            int pt = piece_of(pc);
            if (pt == PAWN) {
                int dir = c == WHITE ? 16 : -16;
                ev_mark_attack(attacked[c], sq + dir - 1);
                ev_mark_attack(attacked[c], sq + dir + 1);
            }
            else if (pt == KNIGHT)
                mobility[c][KNIGHT] += ev_step_attacks(pos, sq, KNIGHT_DIRS, 8, attacked[c], c);
            else if (pt == BISHOP)
                mobility[c][BISHOP] += ev_slider_attacks(pos, sq, BISHOP_DIRS, 4, attacked[c], c);
            else if (pt == ROOK)
                mobility[c][ROOK] += ev_slider_attacks(pos, sq, ROOK_DIRS, 4, attacked[c], c);
            else if (pt == QUEEN) {
                mobility[c][QUEEN] += ev_slider_attacks(pos, sq, BISHOP_DIRS, 4, attacked[c], c);
                mobility[c][QUEEN] += ev_slider_attacks(pos, sq, ROOK_DIRS, 4, attacked[c], c);
            }
            else if (pt == KING)
                ev_step_attacks(pos, sq, KING_DIRS, 8, attacked[c], c);
        }
    for (int c = WHITE; c <= BLACK; ++c) {
        int sg = ev_sign(c), enemy = c ^ 1;
        if (count[c][BISHOP] >= 2) A7F(E7_BISHOP_PAIR, 1);
        for (int pt = KNIGHT; pt <= QUEEN; ++pt) {
            int m = mobility[c][pt];
            if (m > E7_MOB_MAX[pt]) m = E7_MOB_MAX[pt];
            if (m < 0) m = 0;
            A7F(E7_MOB_BASE[pt] + m, 1);
        }
        for (int sq = 0; sq < 128; ++sq) {
            if (!on_board(sq)) { sq += 7; continue; }
            int pc = bb_piece_ref(pos, sq);
            if (!pc || color_of(pc) != c) continue;
            int pt = piece_of(pc);
            int f = file_of(sq), r = ev_relrank(sq, c);
            int s64 = e7_sq64(sq, c);
            mg += sg * W7_PST_MG[pt - 1][s64];
            eg += sg * W7_PST_EG[pt - 1][s64];
            if (pt == PAWN) {
                if (pawns[c][f] > 1) A7F(E7_DOUBLED, 1);
                if ((f == 0 || pawns[c][f - 1] == 0) &&
                    (f == 7 || pawns[c][f + 1] == 0)) A7F(E7_ISOLATED, 1);
                if (ev_is_passed_pawn(pos, c, sq)) {
                    int rr = r < 0 ? 0 : (r > 7 ? 7 : r);
                    A7F(E7_PASSED + rr, 1);
                    int front = sq + (c == WHITE ? 16 : -16);
                    if (on_board(front) && !bb_piece_ref(pos, front)) {
                        A7F(E7_PASS_FREE, 1);
                        A7F(E7_PASS_FREE_R, r);
                    }
                }
                if (pawnAttack[c][sq]) A7F(E7_PAWN_DEF, 1);
                int fr = sq + (c == WHITE ? 16 : -16);
                if (on_board(fr) && bb_piece_ref(pos, fr)) A7F(E7_PAWN_BLOCKED, 1);
            }
            if (pt == ROOK) {
                bool ownPawn = false, enemyPawn = false;
                for (int rr = 0; rr < 8; ++rr) {
                    int x = bb_piece_ref(pos, sq88(f, rr));
                    if (x == make_pc(PAWN, c)) ownPawn = true;
                    if (x == make_pc(PAWN, enemy)) enemyPawn = true;
                }
                if (!ownPawn && !enemyPawn) A7F(E7_ROOK_OPEN, 1);
                else if (!ownPawn) A7F(E7_ROOK_SEMI, 1);
            }
            if (pt == KNIGHT && r >= 3 && r <= 5) {
                int chase = 0, er = rank_of(sq) + (enemy == WHITE ? -1 : 1);
                for (int df = -1; df <= 1; df += 2) {
                    int ef = f + df;
                    if (valid_file(ef) && valid_rank(er) &&
                        bb_piece_ref(pos, sq88(ef, er)) == make_pc(PAWN, enemy)) chase = 1;
                }
                if (!chase) A7F(E7_KNIGHT_OUT, 1);
            }
            if (attacked[c][sq] && !pawnAttack[enemy][sq] && pt != PAWN && pt != KING)
                A7F(E7_PIECE_DEF, 1);
        }
        int cc = 0;
        for (int rr = 2; rr <= 5; ++rr)
            for (int ff = 2; ff <= 5; ++ff)
                if (attacked[c][sq88(ff, rr)]) ++cc;
        A7F(E7_CENTER, cc);
        int ksq = kingSq[c];
        if (ksq != NO_SQ) {
            int f = file_of(ksq), r = rank_of(ksq), ring = 0;
            for (int d : KING_DIRS) {
                int to = ksq + d;
                if (on_board(to) && attacked[enemy][to]) ++ring;
            }
            if (ring > 8) ring = 8;
            A7F(E7_KRING + ring, 1);
            for (int df = -1; df <= 1; ++df) {
                int ff = f + df;
                if (!valid_file(ff)) continue;
                for (int rr = 1; rr <= 2; ++rr) {
                    int pr = c == WHITE ? r + rr : r - rr;
                    if (!valid_rank(pr)) continue;
                    if (bb_piece_ref(pos, sq88(ff, pr)) == make_pc(PAWN, c))
                        A7F(rr == 1 ? E7_KSHIELD1 : E7_KSHIELD2, 1);
                }
            }
            int home = c == WHITE ? 0 : 7;
            if (r != home) A7F(E7_KOFFHOME, 1);
            if (f >= 2 && f <= 5 && r == home) A7F(E7_KCENTRALHOME, 1);
        }
        if (c == WHITE) { if (pos.castle & (WK | WQ)) A7F(E7_CASTLE, 1); }
        else { if (pos.castle & (BK | BQ)) A7F(E7_CASTLE, 1); }
    }
#undef A7F
    int scale = ev_material_scale(count);
    int score = (mg * phase + eg * (24 - phase)) / 24;
    score = score * scale / 128;
    score += pos.sideToMove == WHITE ? 8 : -8;
    return pos.sideToMove == WHITE ? score : -score;
}
struct SearchTTEntry {
    uint64_t key = 0;
    Move move = { 0, 0, 0, 0 };
    int32_t score = 0;
    int8_t depth = 0, flag = 0;
};
// v03a: dynamic transposition table.
// The old version had a fixed 1<<16 TT, which is far too small for real games.
// This keeps the same TT logic, but lets GUI/Arena set Hash in MB.
static constexpr int SEARCH_DEFAULT_HASH_MB = 64;
static constexpr int SEARCH_MIN_HASH_MB = 1;
static constexpr int SEARCH_MAX_HASH_MB = 1024;
static int search_hash_mb = SEARCH_DEFAULT_HASH_MB;
static std::vector<SearchTTEntry> search_tt;
static size_t search_tt_mask = 0;
static SearchTTEntry search_tt_dummy;
static int search_clamp_hash_mb(int mb) {
    if (mb < SEARCH_MIN_HASH_MB) return SEARCH_MIN_HASH_MB;
    if (mb > SEARCH_MAX_HASH_MB) return SEARCH_MAX_HASH_MB;
    return mb;
}
static size_t search_floor_power_of_two(size_t x) {
    if (x < 1) return 1;
    size_t p = 1;
    while (p <= x / 2) p <<= 1;
    return p;
}
static size_t search_tt_entries_for_mb(int mb) {
    mb = search_clamp_hash_mb(mb);
    uint64_t bytes = (uint64_t)mb * 1024ULL * 1024ULL;
    size_t entries = (size_t)(bytes / (uint64_t)sizeof(SearchTTEntry));
    if (entries < 1) entries = 1;
    // Important for speed: keep TT size power-of-two so indexing is key & mask,
    // not a slow runtime modulo/division in every searched node.
    return search_floor_power_of_two(entries);
}
static bool resize_search_tt(int mb) {
    mb = search_clamp_hash_mb(mb);
    size_t entries = search_tt_entries_for_mb(mb);
    try {
        std::vector<SearchTTEntry> fresh(entries);
        search_tt.swap(fresh);
        search_tt_mask = search_tt.empty() ? 0 : (search_tt.size() - 1);
        search_hash_mb = mb;
        return true;
    }
    catch (...) {
        return false;
    }
}
static void clear_search_tt() {
    if (search_tt.empty()) {
        resize_search_tt(search_hash_mb);
        return;
    }
    std::fill(search_tt.begin(), search_tt.end(), SearchTTEntry{});
}
static SearchTTEntry& search_tt_entry(uint64_t key) {
    if (search_tt.empty() && !resize_search_tt(search_hash_mb)) {
        search_tt_dummy = SearchTTEntry{};
        return search_tt_dummy;
    }
    return search_tt[(size_t)key & search_tt_mask];
}
static Move search_killer_moves[2][64];
static int search_history_moves[16][128];
struct SearchPVTable {
    int length = 0;
    Move moves[64] = {};
};
static SearchPVTable search_pv_table[64];
struct SearchLimitsA {
    long long start_time = 0, time_limit = 0;
    bool check_time = false;
    std::atomic<bool> stopped{ false };
    long long nodes = 0;
    // build0.6: zero means no node limit for internal search() calls.
    uint64_t node_limit = 0;
    bool check_nodes = false;
};
static SearchLimitsA search_limits;
// build1.0: soft-limit handoff from uci_go to search(). uci_go sets it for
// clock-based searches (wtime/btime); search() copies it into a local and
// zeroes the global, so bench/tests/movetime searches (which never set it)
// always run with time management OFF. 0 = disabled.
static long long g_tm_soft_ms = 0;
enum SearchStopReason : int {
    SEARCH_STOP_NONE = 0,
    SEARCH_STOP_NODES,
    SEARCH_STOP_MATE,
    SEARCH_STOP_DEPTH,
    SEARCH_STOP_TIME,
    SEARCH_STOP_SOFT_TIME,
    SEARCH_STOP_EXTERNAL
};
static std::atomic<int> search_stop_reason{ SEARCH_STOP_NONE };
static void search_set_reason_if_none(SearchStopReason reason) {
    int expected = SEARCH_STOP_NONE;
    search_stop_reason.compare_exchange_strong(expected, (int)reason);
}
static const char* search_stop_reason_name() {
    switch ((SearchStopReason)search_stop_reason.load()) {
    case SEARCH_STOP_NODES: return "nodes";
    case SEARCH_STOP_MATE: return "mate";
    case SEARCH_STOP_DEPTH: return "depth";
    case SEARCH_STOP_TIME: return "time";
    case SEARCH_STOP_SOFT_TIME: return "softtime";
    case SEARCH_STOP_EXTERNAL: return "stop";
    default: return "unknown";
    }
}
// ============================================================================
// build0.4: telemetry-only instrumentation. Search is single-threaded, so
// these counters are plain (non-atomic) and are reset exactly once at the
// start of each external search() call — never per depth, never inside
// negamax/qsearch, and never before an aspiration re-search.
// ============================================================================
struct SearchTelemetry {
    uint64_t qnodes = 0;
    uint64_t ttProbes = 0;
    uint64_t ttHits = 0;
    uint64_t ttCutoffs = 0;
    uint64_t betaCutoffs = 0;
    uint64_t firstMoveCutoffs = 0;
    uint64_t nullTries = 0;
    uint64_t nullCutoffs = 0;
    uint64_t lmrReductions = 0;
    uint64_t lmrResearches = 0;
    uint64_t aspirationResearches = 0;
    // build0.8: SEE instrumentation. seeCalls counts every search_see()
    // evaluation (ordering + qsearch pruning); seePruned counts captures
    // skipped in qsearch because SEE < 0.
    uint64_t seeCalls = 0;
    uint64_t seePruned = 0;
    // build0.9: futilityPruned counts quiet moves skipped in negamax;
    // deltaPruned counts captures skipped in qsearch by delta pruning.
    uint64_t futilityPruned = 0;
    uint64_t deltaPruned = 0;
    // build1.0: tmSoftStops counts searches ended by the soft time limit
    // (no new iteration started); tmFailLowExtends counts iterations whose
    // soft budget was extended because the score dropped >40cp.
    uint64_t tmSoftStops = 0;
    uint64_t tmFailLowExtends = 0;
};
static SearchTelemetry search_telemetry;
static std::thread g_search_thread;
static std::atomic<bool> g_search_running{ false };
// build0.6: UCI waits for search() to finish resetting its stop state before
// accepting a later "stop", preventing that stop from being overwritten.
static std::atomic<bool> g_search_initialized{ false };
static void stop_search() {
    if (g_search_thread.joinable())
        search_set_reason_if_none(SEARCH_STOP_EXTERNAL);
    search_limits.stopped = true;
    if (g_search_thread.joinable()) {
        g_search_thread.join();
    }
    g_search_running = false;
}
static inline long long search_now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static inline void search_check_limits() {
    // build0.6: node budgets are exact and therefore checked at every
    // counted search entry. Time remains sampled every 2048 nodes.
    if (search_limits.check_nodes &&
        (uint64_t)search_limits.nodes >= search_limits.node_limit) {
        search_set_reason_if_none(SEARCH_STOP_NODES);
        search_limits.stopped = true;
        return;
    }
    if (search_limits.check_time && (search_limits.nodes & 2047) == 0) {
        if (search_now() - search_limits.start_time >= search_limits.time_limit) {
            search_set_reason_if_none(SEARCH_STOP_TIME);
            search_limits.stopped = true;
        }
    }
}
static inline bool search_enter_node() {
    if (search_limits.stopped) return false;
    if (search_limits.nodes == LLONG_MAX) {
        search_set_reason_if_none(SEARCH_STOP_NODES);
        search_limits.stopped = true;
        return false;
    }
    ++search_limits.nodes;
    search_check_limits();
    return !search_limits.stopped;
}
static inline int search_mvv_lva(int attacker, int victim) {
    return search_piece_vals[piece_of(victim)] * 10 - piece_of(attacker);
}
// ============================================================================
// build0.8: SEE — Static Exchange Evaluation, intermediate variant B.
//
// Classic iterative swap algorithm on the 0x88 board:
//   - full attacker discovery for both sides against the target square;
//   - X-RAY SUPPORT: pieces already used in the exchange are marked in
//     removed[] and treated as transparent by the sliding scans, so a
//     rook/queen (on rook lines) or bishop/queen (on diagonals) hiding
//     behind a previous attacker enters the exchange naturally;
//   - NO pin awareness (Level A philosophy, same contract as
//     hash_ep_square() from build0.3): SEE is a cheap local estimate,
//     real legality stays with generateLegalMoves()/search_make_move();
//   - king participation: the king uses search_piece_vals[KING] = 20000,
//     so a king "capture" that gets recaptured folds to a huge loss and
//     is discarded by the minimax fold — the standard safe treatment.
//
// see_least_valuable_attacker(): returns the square of the least valuable
// attacker of `side` against `to`, considering removed[] squares as empty
// (that is exactly what produces x-rays). Scan order = ascending value:
// pawn, knight, bishop, rook, queen, king.
// ============================================================================
static int see_least_valuable_attacker_0x88(const Position& p, const bool removed[128],
    int to, int side) {
    // Pawns: a pawn of `side` attacks `to` from one rank behind (relative
    // to its own pushing direction), files +/-1.
    const int fwd = (side == WHITE) ? 16 : -16;
    for (int df = -1; df <= 1; df += 2) {
        int s = to - fwd + df;
        if (on_board(s) && !removed[s] && bb_piece_ref(p, s) == make_pc(PAWN, side))
            return s;
    }
    for (int d : KNIGHT_DIRS) {
        int s = to + d;
        if (on_board(s) && !removed[s] && bb_piece_ref(p, s) == make_pc(KNIGHT, side))
            return s;
    }
    // Sliders: removed[] squares are transparent — this is the x-ray.
    for (int d : BISHOP_DIRS) {
        for (int s = to + d; on_board(s); s += d) {
            if (removed[s]) continue;
            int pc = bb_piece_ref(p, s);
            if (!pc) continue;
            if (pc == make_pc(BISHOP, side)) return s;
            break;
        }
    }
    for (int d : ROOK_DIRS) {
        for (int s = to + d; on_board(s); s += d) {
            if (removed[s]) continue;
            int pc = bb_piece_ref(p, s);
            if (!pc) continue;
            if (pc == make_pc(ROOK, side)) return s;
            break;
        }
    }
    for (int d : BISHOP_DIRS) {
        for (int s = to + d; on_board(s); s += d) {
            if (removed[s]) continue;
            int pc = bb_piece_ref(p, s);
            if (!pc) continue;
            if (pc == make_pc(QUEEN, side)) return s;
            break;
        }
    }
    for (int d : ROOK_DIRS) {
        for (int s = to + d; on_board(s); s += d) {
            if (removed[s]) continue;
            int pc = bb_piece_ref(p, s);
            if (!pc) continue;
            if (pc == make_pc(QUEEN, side)) return s;
            break;
        }
    }
    for (int d : KING_DIRS) {
        int s = to + d;
        if (on_board(s) && !removed[s] && bb_piece_ref(p, s) == make_pc(KING, side))
            return s;
    }
    return NO_SQ;
}
// search_see(): static exchange value of the capture `mv` in centipawns,
// from the perspective of the side making the move. >= 0 means the capture
// does not lose material by the swap-off estimate.
static int search_see_0x88(const Position& pos, const Move& mv) {
    search_telemetry.seeCalls++;
    const int to = mv.to;
    bool removed[128] = {};
    int gain[36];
    int side = color_of(bb_piece_ref(pos, mv.from));
    // First capture: what is being taken off `to`.
    if (mv.flags & F_EP) {
        gain[0] = search_piece_vals[PAWN];
        // The captured pawn is NOT on `to` for en passant.
        removed[to + (side == WHITE ? -16 : 16)] = true;
    }
    else {
        gain[0] = bb_piece_ref(pos, to) ? search_piece_vals[piece_of(bb_piece_ref(pos, to))] : 0;
    }
    // The mover now occupies `to`; its origin square becomes transparent.
    removed[mv.from] = true;
    int occupier_val = search_piece_vals[piece_of(bb_piece_ref(pos, mv.from))];
    side ^= 1;
    int d = 0;
    while (d + 1 < 36) {
        int from = see_least_valuable_attacker_0x88(pos, removed, to, side);
        if (from == NO_SQ) break;
        ++d;
        gain[d] = occupier_val - gain[d - 1];
        occupier_val = search_piece_vals[piece_of(bb_piece_ref(pos, from))];
        removed[from] = true;
        side ^= 1;
    }
    while (d > 0) {
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
        --d;
    }
    return gain[0];
}

// Stage 6 SEE attacker selection. The bitboard attack masks provide the
// candidate set after removed exchange pieces are made transparent. The final
// selection follows the old 0x88 direction order so equal-valued attackers
// preserve the original deterministic SEE behavior.
static int bb_first_see_attacker(const Position& p, Bitboard removed,
    int to, int side) {
    const int target64 = bb_sq_from_sq88(to);
    const Bitboard live = ~removed;
    const Bitboard target = bb_square(target64);

    Bitboard candidates = p.bbPieces[side][PAWN] & live;
    int from = bb_first_pawn_attacker(candidates, target64, side);
    if (from != NO_SQ) return from;

    candidates = p.bbPieces[side][KNIGHT] & live &
        bb_knight_attack_mask(target64);
    from = bb_first_candidate_on_dirs(candidates, target64, KNIGHT_DIRS, 8);
    if (from != NO_SQ) return from;

    const Bitboard occupied = p.bbOccupiedAll & live;
    const Bitboard bishop_rays = bb_bishop_attack_mask(target64, occupied);
    const Bitboard rook_rays = bb_rook_attack_mask(target64, occupied);

    candidates = bishop_rays & p.bbPieces[side][BISHOP] & live;
    from = bb_first_candidate_on_dirs(candidates, target64, BISHOP_DIRS, 4);
    if (from != NO_SQ) return from;

    candidates = rook_rays & p.bbPieces[side][ROOK] & live;
    from = bb_first_candidate_on_dirs(candidates, target64, ROOK_DIRS, 4);
    if (from != NO_SQ) return from;

    candidates = bishop_rays & p.bbPieces[side][QUEEN] & live;
    from = bb_first_candidate_on_dirs(candidates, target64, BISHOP_DIRS, 4);
    if (from != NO_SQ) return from;

    candidates = rook_rays & p.bbPieces[side][QUEEN] & live;
    from = bb_first_candidate_on_dirs(candidates, target64, ROOK_DIRS, 4);
    if (from != NO_SQ) return from;

    candidates = p.bbPieces[side][KING] & live &
        bb_king_attack_mask(target64);
    return bb_first_candidate_on_dirs(candidates, target64, KING_DIRS, 8);
}

// Bitboard SEE path. It preserves the original build0.8 contract: x-rays are
// allowed through removed pieces and pins are intentionally not considered.
static int search_see(const Position& pos, const Move& mv) {
    search_telemetry.seeCalls++;
    const int to = mv.to;
    Bitboard removed = 0;
    int gain[36];
    int side = color_of(bb_piece_ref(pos, mv.from));
    const Bitboard from_bit = bb_square(bb_sq_from_sq88(mv.from));
    if (mv.flags & F_EP) {
        gain[0] = search_piece_vals[PAWN];
        const int captured = to + (side == WHITE ? -16 : 16);
        removed |= bb_square(bb_sq_from_sq88(captured));
    }
    else {
        gain[0] = bb_piece_ref(pos, to) ? search_piece_vals[piece_of(bb_piece_ref(pos, to))] : 0;
    }
    removed |= from_bit;
    int occupier_val = search_piece_vals[piece_of(bb_piece_ref(pos, mv.from))];
    side ^= 1;
    int d = 0;
    while (d + 1 < 36) {
        int from = bb_first_see_attacker(pos, removed, to, side);
        if (from == NO_SQ) break;
        ++d;
        gain[d] = occupier_val - gain[d - 1];
        occupier_val = search_piece_vals[piece_of(bb_piece_ref(pos, from))];
        removed |= bb_square(bb_sq_from_sq88(from));
        side ^= 1;
    }
    while (d > 0) {
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
        --d;
    }
    return gain[0];
}

static int search_score_move(const Position& pos, Move mv, int ply, Move tt_move) {
    if (move_equal(mv, tt_move)) return 100000;
    int pc = bb_piece_ref(pos, mv.from), cap = bb_piece_ref(pos, mv.to);
    // build0.8: captures are split by SEE into good (>= 0, above killers)
    // and bad (< 0, below killers but above quiet history moves).
    // Promotions (incl. capture-promotions) are forcing and never demoted.
    if ((mv.flags & F_EP) || cap != EMPTY) {
        int victim = (mv.flags & F_EP)
            ? make_pc(PAWN, pos.sideToMove == WHITE ? BLACK : WHITE)
            : cap;
        int base = search_mvv_lva(pc, victim);
        if (!(mv.flags & F_PROMO) && search_see(pos, mv) < 0)
            return 10000 + base; // bad capture: below killers (30000/20000)
        return 50000 + base;
    }
    if (mv.flags & F_PROMO) return 40000 + search_piece_vals[piece_of(mv.promo)];
    if (ply < 64 && move_equal(search_killer_moves[0][ply], mv)) return 30000;
    if (ply < 64 && move_equal(search_killer_moves[1][ply], mv)) return 20000;
    return search_history_moves[pc][mv.to];
}
static void search_sort_moves(const Position& pos, MoveList& list, int ply, Move tt_move) {
    int scores[MAX_MOVES];
    for (int i = 0; i < list.count; ++i) scores[i] = search_score_move(pos, list.m[i], ply, tt_move);
    for (int i = 0; i < list.count - 1; ++i) {
        int best_idx = i;
        for (int j = i + 1; j < list.count; ++j)
            if (scores[j] > scores[best_idx]) best_idx = j;
        std::swap(list.m[i], list.m[best_idx]);
        std::swap(scores[i], scores[best_idx]);
    }
}
static void search_generate_moves(const Position& pos, MoveList& list, bool captures_only = false) {
    // ВАЖНО ДЛЯ СКОРОСТИ:
    // В поиске не надо заранее строить legal moves через generateLegalMoves(),
    // потому что generateLegalMoves() уже делает make/unmake для проверки шаха.
    // Потом search_negamax() всё равно снова делает search_make_move().
    // Поэтому тут генерируем псевдолегальные ходы, а нелегальные отсекаем
    // в search_make_move(). Это уменьшает двойную работу.
    MoveList pseudo;
    generateMoves(pos, pseudo);
    list.clear();
    for (int i = 0; i < pseudo.count; ++i) {
        // В quiescence обычно нужны взятия. Превращения тоже считаем forcing-ходами,
        // даже если это тихое превращение без взятия.
        if (!captures_only || (pseudo.m[i].flags & (F_CAPTURE | F_PROMO))) {
            list.add(pseudo.m[i].from, pseudo.m[i].to, pseudo.m[i].promo, pseudo.m[i].flags);
        }
    }
}
// search_make_move/search_unmake_move теперь опционально принимают
// RepetitionStack*. qsearch их не передаёт (nullptr по умолчанию) —
// репутационная история там никогда не запрашивалась и раньше, так что
// поведение не меняется. search_negamax передаёт указатель на свою копию
// стека явно.
static bool search_make_move(Position& pos, const Move& mv, Undo& undo, RepetitionStack* rep = nullptr) {
    const int us = pos.sideToMove;
    // Так как поиск теперь получает псевдолегальные ходы, рокировку надо
    // проверять отдельно ДО makeMove(): обычная проверка после makeMove()
    // ловит только шах на конечном поле, но не ловит рокировку из шаха
    // и не ловит проход короля через битое поле.
    if (mv.flags & F_CASTLE) {
        if (inCheck(pos, us))
            return false;
        int through = (mv.from + mv.to) / 2;
        if (isSquareAttacked(pos, through, us ^ 1))
            return false;
        if (isSquareAttacked(pos, mv.to, us ^ 1))
            return false;
    }
    makeMove(pos, mv, undo);
    if (inCheck(pos, us)) {
        unmakeMove(pos, mv, undo);
        return false;
    }
    if (rep) rep->push(pos.key);
    return true;
}
static void search_unmake_move(Position& pos, const Move& mv, const Undo& undo, RepetitionStack* rep = nullptr) {
    if (rep) rep->pop();
    unmakeMove(pos, mv, undo);
}
static int search_qsearch(Position& pos, int alpha, int beta, int depth) {
    // build0.6: do not count a fresh qsearch entry after an external stop.
    if (search_limits.stopped) return 0;
    search_telemetry.qnodes++;
    if (!search_enter_node()) return 0;
    const bool in_check = inCheck(pos, pos.sideToMove);
    // build0.9: stand_pat is hoisted out of the else-branch so delta pruning
    // can see it inside the move loop (valid only when !in_check).
    int stand_pat = 0;
    // build0.9: lazy non-pawn-material scan for the delta-pruning endgame
    // guard. Computed at most once per qsearch node, and only if a capture
    // actually reaches the delta test. -1 = not computed yet.
    int qs_npm = -1;
    MoveList list;
    if (in_check) {
        // Build the same ordered legal-evasion list in place: the parent is
        // already in check, so castling is impossible. This avoids a Position
        // copy and lets the recursion skip a second legality test.
        MoveList pseudo;
        generateMoves(pos, pseudo);
        list.clear();
        const int us = pos.sideToMove;
        for (int i = 0; i < pseudo.count; ++i) {
            const Move mv = pseudo.m[i];
            if (mv.flags & F_CASTLE) continue;
            Undo legality_undo;
            makeMove(pos, mv, legality_undo);
            const bool legal = !inCheck(pos, us);
            unmakeMove(pos, mv, legality_undo);
            if (legal) list.add(mv.from, mv.to, mv.promo, mv.flags);
        }
        if (list.count == 0)
            return -SEARCH_MATE_SCORE + depth;
    }
    else {
        stand_pat = evaluate(pos);
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;
        search_generate_moves(pos, list, true);
    }
    search_sort_moves(pos, list, depth, Move{ 0,0,0,0 });
    for (int i = 0; i < list.count; ++i) {
        // build0.9: delta pruning — BEFORE the (costly) SEE call. If even
        // capturing the victim plus a safety margin cannot lift stand_pat
        // to alpha, the capture is futile. Never applied in check or to
        // promotions; disabled in low-material endgames (npm <= 1300),
        // where "hopeless" captures routinely hide fortress/stalemate
        // resources deeper in the tree.
        if (!in_check && (list.m[i].flags & F_CAPTURE) &&
            !(list.m[i].flags & F_PROMO)) {
            int victim = (list.m[i].flags & F_EP)
                ? PAWN
                : piece_of(bb_piece_ref(pos, list.m[i].to));
            if (stand_pat + search_piece_vals[victim] + SEARCH_DELTA_MARGIN <= alpha) {
                if (qs_npm < 0) {
                    qs_npm = 0;
                    for (int sq = 0; sq < 128; ++sq) {
                        if (sq & 0x88) continue;
                        int pc = bb_piece_ref(pos, sq);
                        if (pc == EMPTY) continue;
                        int pt = piece_of(pc);
                        if (pt != PAWN && pt != KING)
                            qs_npm += search_piece_vals[pt];
                    }
                }
                if (qs_npm > SEARCH_DELTA_NPM_MIN) {
                    search_telemetry.deltaPruned++;
                    continue;
                }
            }
        }
        // build0.8: SEE pruning of losing captures — ONLY when not in check
        // (the in-check branch searches full legal evasions and is never
        // pruned). Promotions are forcing and are never pruned either.
        if (!in_check && (list.m[i].flags & F_CAPTURE) &&
            !(list.m[i].flags & F_PROMO) &&
            search_see(pos, list.m[i]) < 0) {
            search_telemetry.seePruned++;
            continue;
        }
        Undo undo;
        if (in_check) {
            // list contains only legal evasions, already checked above.
            makeMove(pos, list.m[i], undo);
        }
        else if (!search_make_move(pos, list.m[i], undo)) {
            continue;
        }
        int score = -search_qsearch(
            pos,
            -beta,
            -alpha,
            depth + 1
        );
        if (in_check) unmakeMove(pos, list.m[i], undo);
        else search_unmake_move(pos, list.m[i], undo);
        if (search_limits.stopped)
            return 0;
        if (score >= beta)
            return beta;
        if (score > alpha)
            alpha = score;
    }
    return alpha;
}
static bool is_draw_by_rules(const Position& pos) {
    if (pos.halfmove >= 100) return true;
    int piece_count = 0;
    int knight_count = 0;
    int bishop_count = 0;
    for (int sq = 0; sq < 128; ++sq) {
        if (!on_board(sq)) continue;
        int pc = bb_piece_ref(pos, sq);
        if (pc == EMPTY) continue;
        int pt = piece_of(pc);
        if (pt != KING) piece_count++;
        if (pt == KNIGHT) knight_count++;
        else if (pt == BISHOP) bishop_count++;
    }
    if (piece_count == 0) return true;
    if (piece_count == 1 && (knight_count == 1 || bishop_count == 1)) return true;
    return false;
}
// Теперь принимает явный стек вместо обращения к глобалу.
static bool is_repetition(const Position& pos, const RepetitionStack& rep) {
    int end = rep.count - 3;
    // FIX (build0.1): ближайший индекс, который в принципе может повторять
    // текущую позицию — это позиция сразу ПОСЛЕ последнего необратимого хода,
    // т.е. (последний индекс) - pos.halfmove = (rep.count - 1) - pos.halfmove.
    // Старая формула (rep.count - pos.halfmove) была сдвинута на единицу и
    // молча исключала эту граничную позицию (в т.ч. root) из сканирования.
    int limit = rep.count - pos.halfmove - 1;
    if (limit < 0) limit = 0;
    for (int i = end; i >= limit; i -= 2) {
        if (rep.keys[i] == pos.key)
            return true;
    }
    return false;
}
static int search_negamax(Position& pos, int alpha, int beta, int depth, int ply, RepetitionStack& rep, bool do_null = true) {
    // build0.6: guard before the ply-64 handoff prevents a post-stop qnode.
    if (search_limits.stopped) return 0;
    if (ply >= 64) return search_qsearch(pos, alpha, beta, ply);
    // FIX: запоминаем исходный alpha. В конце TT-флаг нужно определять
    // относительно начального окна, а не относительно уже измененного alpha.
    const int alpha_orig = alpha;
    search_pv_table[ply].length = 0;
    if (!search_enter_node()) return 0;
    bool in_check = inCheck(pos, pos.sideToMove);
    // build0.2 FIX 2: мат должен иметь приоритет над 50-move draw и над
    // repetition. Если draw-условие сработало, но сторона находится в
    // шахе и у неё нет легальных ходов — это мат, а не ничья. Легальные
    // ходы генерируются дополнительно только в этом (редком) случае.
    if (ply > 0 &&
        (is_draw_by_rules(pos) || is_repetition(pos, rep))) {
        if (!in_check)
            return 0;
        MoveList legal;
        generateLegalMoves(pos, legal);
        if (legal.count == 0)
            return -SEARCH_MATE_SCORE + ply;
        return 0;
    }
    if (in_check) depth++;
    if (depth <= 0) return search_qsearch(pos, alpha, beta, ply);
    uint64_t hash = pos.key;
    search_telemetry.ttProbes++;
    SearchTTEntry& entry = search_tt_entry(hash);
    Move tt_move = { 0,0,0,0 };
    if (entry.key == hash) {
        search_telemetry.ttHits++;
        tt_move = entry.move;
        int score = entry.score;
        if (score > SEARCH_MATE_SCORE - 100) score -= ply;
        else if (score < -SEARCH_MATE_SCORE + 100) score += ply;
        // Не делаем TT-cutoff на корне (ply == 0), иначе Arena может видеть
        // пустой/короткий PV на малых глубинах: позиция возвращается из TT
        // раньше, чем построена principal variation. TT-ход всё равно используется
        // выше как tt_move для сортировки ходов.
        if (ply > 0 && entry.depth >= depth) {
            if (entry.flag == 0) { search_telemetry.ttCutoffs++; return score; }
            if (entry.flag == 1 && score <= alpha) { search_telemetry.ttCutoffs++; return alpha; }
            if (entry.flag == 2 && score >= beta) { search_telemetry.ttCutoffs++; return beta; }
        }
    }
    if (do_null && !in_check && depth >= 3 && ply > 0) {
        bool major = false;
        for (int sq = 0; sq < 128; ++sq) {
            if (!(sq & 0x88) && bb_piece_ref(pos, sq) != EMPTY &&
                piece_of(bb_piece_ref(pos, sq)) != PAWN && piece_of(bb_piece_ref(pos, sq)) != KING) {
                if (color_of(bb_piece_ref(pos, sq)) == pos.sideToMove) { major = true; break; }
            }
        }
        if (major) {
            // build0.3: the manual null-move hash patch used to XOR the RAW
            // zEp[temp_ep] in/out directly. That desynced it from
            // computeHash()/makeMove(), which now both normalize the ep
            // component through hash_ep_square(). We replicate the same
            // normalization here, and additionally save/restore the exact
            // key via a plain snapshot so the position is bit-for-bit
            // restored afterwards (no accumulated XOR drift possible).
            const uint64_t key_before_null = pos.key;
            const int saved_ep = pos.ep;
            const int old_ep_component = hash_ep_square(pos);
            pos.sideToMove ^= 1;
            pos.key ^= zSide;
            pos.ep = NO_SQ;
            const int new_ep_component = hash_ep_square(pos);
            pos.key ^= zEp[old_ep_component] ^ zEp[new_ep_component];
            search_telemetry.nullTries++;
            int score = -search_negamax(pos, -beta, -beta + 1, depth - 3, ply + 1, rep, false);
            pos.ep = saved_ep;
            pos.sideToMove ^= 1;
            pos.key = key_before_null;
            if (search_limits.stopped) return 0;
            if (score >= beta) { search_telemetry.nullCutoffs++; return beta; }
        }
    }
    // build0.9: futility pre-check. A node is "futile" when it is a
    // frontier/pre-frontier node (depth <= 2) inside a null window and the
    // static eval plus a depth-scaled margin still cannot reach alpha.
    // Guards: never in check, never at the root, never in mate-score
    // windows (mate search must stay exact). evaluate() is called at most
    // once here and only for candidate nodes.
    bool futile_node = false;
    if (depth <= 2 && !in_check && ply > 0 && beta - alpha == 1 &&
        alpha > -SEARCH_MATE_SCORE + 200 && beta < SEARCH_MATE_SCORE - 200) {
        if (evaluate(pos) + SEARCH_FUTILITY_MARGIN[depth] <= alpha)
            futile_node = true;
    }
    MoveList list;
    search_generate_moves(pos, list);
    search_sort_moves(pos, list, ply, tt_move);
    int legal = 0, best_score = -SEARCH_INFINITE_SCORE;
    Move best_move = { 0,0,0,0 };
    for (int i = 0; i < list.count; ++i) {
        Move mv = list.m[i]; Undo undo;
        if (!search_make_move(pos, mv, undo, &rep)) continue;
        legal++;
        // build0.9: in-loop futility pruning. In a futile node, quiet
        // non-promotion moves that do not give check are skipped. The
        // first legal move is always searched (legal > 1) so the node
        // keeps a valid score/TT move even when everything is futile.
        if (futile_node && legal > 1 &&
            !(mv.flags & (F_CAPTURE | F_PROMO)) &&
            !inCheck(pos, pos.sideToMove)) {
            search_telemetry.futilityPruned++;
            search_unmake_move(pos, mv, undo, &rep);
            continue;
        }
        int score;
        if (depth >= 3 && legal > 4 && !(mv.flags & F_CAPTURE) && !in_check && !inCheck(pos, pos.sideToMove)) {
            search_telemetry.lmrReductions++;
            score = -search_negamax(pos, -alpha - 1, -alpha, depth - 2, ply + 1, rep);
            if (score > alpha && score < beta) {
                search_telemetry.lmrResearches++;
                score = -search_negamax(pos, -beta, -alpha, depth - 1, ply + 1, rep);
            }
        }
        else {
            score = -search_negamax(pos, -beta, -alpha, depth - 1, ply + 1, rep);
        }
        search_unmake_move(pos, mv, undo, &rep);
        if (search_limits.stopped) return 0;
        if (score > best_score) { best_score = score; best_move = mv; }
        if (score >= beta) {
            search_telemetry.betaCutoffs++;
            if (legal == 1) search_telemetry.firstMoveCutoffs++;
            if (!(mv.flags & F_CAPTURE)) {
                if (ply < 64) {
                    search_killer_moves[1][ply] = search_killer_moves[0][ply];
                    search_killer_moves[0][ply] = mv;
                }
                search_history_moves[bb_piece_ref(pos, mv.from)][mv.to] += depth * depth;
            }
            entry.key = hash; entry.move = mv; entry.depth = depth; entry.flag = 2;
            entry.score = (best_score > SEARCH_MATE_SCORE - 100) ? best_score + ply
                : (best_score < -SEARCH_MATE_SCORE + 100) ? best_score - ply : best_score;
            return beta;
        }
        if (score > alpha) {
            alpha = score;
            search_pv_table[ply].moves[0] = mv;
            int next_len = (ply + 1 < 64) ? search_pv_table[ply + 1].length : 0;
            if (next_len > 63) next_len = 63;
            for (int j = 0; j < next_len; ++j)
                search_pv_table[ply].moves[j + 1] = search_pv_table[ply + 1].moves[j];
            search_pv_table[ply].length = next_len + 1;
        }
    }
    if (legal == 0) return in_check ? -SEARCH_MATE_SCORE + ply : 0;
    entry.key = hash; entry.move = best_move; entry.depth = depth;
    // FIX: раньше было (best_score > alpha), но alpha уже изменен.
    // Поэтому EXACT почти никогда не записывался. Правильно сравнивать
    // с alpha_orig — исходной границей окна на входе в функцию.
    entry.flag = (best_score <= alpha_orig) ? 1 : 0; // 1 = UPPER, 0 = EXACT
    entry.score = (best_score > SEARCH_MATE_SCORE - 100) ? best_score + ply
        : (best_score < -SEARCH_MATE_SCORE + 100) ? best_score - ply : best_score;
    return alpha;
}
// ============================================================================
// build0.2 P0 TESTS
//
// Постоянные регрессионные проверки для P0-исправлений. Не заменяют внешнее
// тестирование, но гарантируют, что базовые инварианты (мат важнее draw-return,
// draw rules не считают K+N vs K+N мёртвой позицией) не откатятся незаметно
// в будущих правках.
// ============================================================================
static void run_p0_tests() {
    int pass = 0;
    int fail = 0;
    auto result = [&](bool ok, const char* name) {
        if (ok) {
            ++pass;
            std::printf("info string P0 %s OK\n", name);
        }
        else {
            ++fail;
            std::printf("info string P0 %s FAIL\n", name);
        }
        };
    Position p;
    RepetitionStack rep;
    bool ok = parse_fen(
        p,
        "7k/5KQ1/8/8/8/8/8/8 b - - 100 60"
    );
    rep.count = 0;
    if (ok) rep.push(p.key);
    search_limits.stopped = false;
    search_limits.check_time = false;
    search_limits.check_nodes = false;
    search_limits.node_limit = 0;
    search_limits.nodes = 0;
    int mate_score = ok
        ? search_negamax(
            p,
            -SEARCH_INFINITE_SCORE,
            SEARCH_INFINITE_SCORE,
            1,
            1,
            rep,
            false
        )
        : 0;
    result(
        ok && mate_score == -SEARCH_MATE_SCORE + 1,
        "mate_before_50move"
    );
    ok = parse_fen(
        p,
        "7k/8/8/8/8/8/6R1/6K1 b - - 100 60"
    );
    rep.count = 0;
    if (ok) rep.push(p.key);
    search_limits.stopped = false;
    search_limits.check_time = false;
    search_limits.check_nodes = false;
    search_limits.node_limit = 0;
    search_limits.nodes = 0;
    int draw_score = ok
        ? search_negamax(
            p,
            -SEARCH_INFINITE_SCORE,
            SEARCH_INFINITE_SCORE,
            1,
            1,
            rep,
            false
        )
        : 1;
    result(ok && draw_score == 0, "fifty_move_draw");
    ok = parse_fen(
        p,
        "7k/5n2/8/8/8/8/2N5/K7 w - - 0 1"
    );
    result(
        ok && !is_draw_by_rules(p),
        "knight_vs_knight_not_dead"
    );
    ok = parse_fen(
        p,
        "7k/8/8/8/8/8/2N5/K7 w - - 0 1"
    );
    result(
        ok && is_draw_by_rules(p),
        "knight_vs_king_dead"
    );
    std::printf(
        "info string P0 TESTS DONE pass %d fail %d\n",
        pass,
        fail
    );
    std::fflush(stdout);
}
// ============================================================================
// build0.3 EP TESTS
//
// Regression coverage for canonical en-passant Zobrist hashing (Level A).
// See hash_ep_square() for the exact normalization contract. These tests are
// intentionally independent of search/TT and only touch parse_fen,
// computeHash, makeMove, generateMoves/generateLegalMoves and
// hash_ep_square directly, plus a standalone replica of the null-move key
// patch pattern used inside search_negamax().
// ============================================================================
static void run_ep_tests() {
    int pass = 0;
    int fail = 0;
    auto result = [&](bool ok, const char* name) {
        if (ok) {
            ++pass;
            std::printf("info string EP %s OK\n", name);
        }
        else {
            ++fail;
            std::printf("info string EP %s FAIL\n", name);
        }
        };
    // A. double_without_candidate_canonical
    {
        Position p, q; Move m{}; Undo u;
        bool ok = parse_fen(p, "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1") && getmv(p, "e2e4", m);
        if (ok) {
            makeMove(p, m, u);
            ok = ok && (p.key == computeHash(p));
            ok = ok && parse_fen(q, "4k3/8/8/8/4P3/8/8/4K3 b - - 0 1");
            ok = ok && (p.key == q.key);
        }
        result(ok, "double_without_candidate_canonical");
    }
    // B. double_with_candidate_hashed
    {
        Position p, q, r; Move m{}; Undo u;
        bool ok = parse_fen(p, "4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1") && getmv(p, "e2e4", m);
        if (ok) {
            makeMove(p, m, u);
            ok = ok && (p.key == computeHash(p));
            ok = ok && parse_fen(q, "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1");
            ok = ok && (p.key == q.key);
            ok = ok && parse_fen(r, "4k3/8/8/8/3pP3/8/8/4K3 b - - 0 1");
            ok = ok && (p.key != r.key);
        }
        result(ok, "double_with_candidate_hashed");
    }
    // C. malformed_ep_no_move
    {
        const char* fen = "4k3/8/8/4P3/8/8/8/4K3 w - d6 0 1";
        bool ok = !has(fen, "e5d6") && !has_legal(fen, "e5d6");
        result(ok, "malformed_ep_no_move");
    }
    // D. malformed_ep_not_hashed
    {
        Position p, q;
        bool ok = parse_fen(p, "4k3/8/8/4P3/8/8/8/4K3 w - d6 0 1")
            && parse_fen(q, "4k3/8/8/4P3/8/8/8/4K3 w - - 0 1");
        ok = ok && (p.key == q.key) && (hash_ep_square(p) == NO_SQ);
        result(ok, "malformed_ep_not_hashed");
    }
    // E. pinned_ep_level_a_boundary
    // Documented Level A limitation: hash_ep_square() only checks geometry
    // and captured-pawn existence, not pin legality. The capture itself is
    // still correctly rejected as illegal by generateLegalMoves() because it
    // would expose the White king to the rook on a5. This is intentional and
    // not considered a Level A failure.
    {
        const char* fen = "8/8/8/r2pP2K/8/8/8/8 w - d6 0 1";
        Position p;
        bool ok = parse_fen(p, fen);
        ok = ok && (hash_ep_square(p) == sq_of("d6"));
        ok = ok && !has_legal(fen, "e5d6");
        result(ok, "pinned_ep_level_a_boundary");
    }
    // F. null_ep_roundtrip
    // Replicates the exact key-patch pattern used in search_negamax()'s
    // manual null move, standalone (no search/TT/thread involvement).
    {
        Position p;
        bool ok = parse_fen(p, "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1");
        if (ok) {
            const uint64_t original_key = p.key;
            ok = ok && (original_key == computeHash(p));
            const uint64_t key_before_null = p.key;
            const int saved_ep = p.ep;
            const int old_ep_component = hash_ep_square(p);
            p.sideToMove ^= 1;
            p.key ^= zSide;
            p.ep = NO_SQ;
            const int new_ep_component = hash_ep_square(p);
            p.key ^= zEp[old_ep_component] ^ zEp[new_ep_component];
            ok = ok && (p.key == computeHash(p));
            p.ep = saved_ep;
            p.sideToMove ^= 1;
            p.key = key_before_null;
            ok = ok && (p.key == original_key);
            ok = ok && (p.key == computeHash(p));
        }
        else {
            ok = false;
        }
        result(ok, "null_ep_roundtrip");
    }
    std::printf("info string EP TESTS DONE pass %d fail %d\n", pass, fail);
    std::fflush(stdout);
}
// ============================================================================
// build0.8 SEE TESTS
//
// Regression coverage for search_see() (intermediate variant B: swap with
// x-ray, no pins). Pure unit tests: parse_fen + generateMoves + search_see,
// no search/TT/thread involvement. Expected values use search_piece_vals
// (P=100, N=320, B=330, R=500, Q=900, K=20000).
// ============================================================================
static void run_see_tests() {
    int pass = 0;
    int fail = 0;
    auto result = [&](bool ok, const char* name) {
        if (ok) {
            ++pass;
            std::printf("info string SEE %s OK\n", name);
        }
        else {
            ++fail;
            std::printf("info string SEE %s FAIL\n", name);
        }
        };
    auto see_of = [](const char* fen, const char* mv, bool& ok, int& out) {
        Position p; Move m{};
        ok = parse_fen(p, fen) && getmv(p, mv, m);
        if (ok) out = search_see(p, m);
        };
    bool ok; int v;
    // 1. PxP, target undefended: clean +100.
    see_of("7k/8/8/3p4/4P3/8/8/7K w - - 0 1", "e4d5", ok, v);
    result(ok && v == 100, "pxp_undefended_plus100");
    // 2. PxP, target defended by a pawn: even trade, exactly 0.
    see_of("7k/8/2p5/3p4/4P3/8/8/7K w - - 0 1", "e4d5", ok, v);
    result(ok && v == 0, "pxp_defended_zero");
    // 3. QxP, pawn defended by a pawn: queen is lost for a pawn, -800.
    see_of("7k/8/2p5/3p4/8/8/8/3Q3K w - - 0 1", "d1d5", ok, v);
    result(ok && v == -800, "qxp_defended_minus800");
    // 4. X-RAY: white rook battery d1+d2 vs black rooks d5+d8.
    //    RxR, RxR, RxR ends +500 — but ONLY if the d1 rook behind the d2
    //    rook is seen through the vacated square. A no-x-ray SEE returns 0.
    see_of("3r3k/8/8/3r4/8/8/3R4/3R3K w - - 0 1", "d2d5", ok, v);
    result(ok && v == 500, "xray_rook_battery_plus500");
    // 5. King "captures" a defended pawn: recapture of the king folds to a
    //    huge loss, so SEE must be negative (king value treatment).
    see_of("7k/8/5p2/4p3/3K4/8/8/8 w - - 0 1", "d4e5", ok, v);
    result(ok && v < 0, "king_takes_defended_negative");
    // 6. En passant: captured pawn sits BESIDE the target square, +100.
    see_of("7k/8/8/3pP3/8/8/8/7K w - d6 0 1", "e5d6", ok, v);
    result(ok && v == 100, "ep_capture_plus100");
    // 7. Ordering integration: a losing capture must be scored below
    //    killers (i.e. below 20000), a winning one above 30000.
    {
        Position p; Move m{};
        bool o2 = parse_fen(p, "7k/8/2p5/3p4/8/8/8/3Q3K w - - 0 1")
            && getmv(p, "d1d5", m);
        int sc = o2 ? search_score_move(p, m, 0, Move{ 0,0,0,0 }) : 999999;
        result(o2 && sc < 20000, "ordering_bad_capture_below_killers");
    }
    {
        Position p; Move m{};
        bool o2 = parse_fen(p, "7k/8/8/3p4/4P3/8/8/7K w - - 0 1")
            && getmv(p, "e4d5", m);
        int sc = o2 ? search_score_move(p, m, 0, Move{ 0,0,0,0 }) : -1;
        result(o2 && sc > 30000, "ordering_good_capture_above_killers");
    }
    // Stage 6 differential test: every legal capture in representative
    // positions must produce the same value through the new bitboard SEE and
    // the preserved 0x88 reference implementation.
    bool differential = true;
    const char* diff_fens[] = {
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "3r3k/8/8/3r4/8/8/3R4/3R3K w - - 0 1",
        "7k/8/2p5/3p4/8/8/8/3Q3K w - - 0 1",
        "7k/8/8/3pP3/8/8/8/7K w - d6 0 1",
        "8/P6p/1p6/3pPp2/8/8/p6P/4K2k w - f6 0 1",
    };
    int compared = 0;
    for (const char* fen : diff_fens) {
        Position p;
        if (!parse_fen(p, fen)) { differential = false; break; }
        MoveList ml;
        generateLegalMoves(p, ml);
        for (int i = 0; i < ml.count; ++i) {
            if (!(ml.m[i].flags & F_CAPTURE)) continue;
            const int old_value = search_see_0x88(p, ml.m[i]);
            const int new_value = search_see(p, ml.m[i]);
            ++compared;
            if (old_value != new_value) {
                differential = false;
                break;
            }
        }
        if (!differential) break;
    }
    result(differential && compared > 0, "bitboard_see_differential");
    std::printf("info string SEE differential_captures %d\n", compared);
    std::printf("info string SEE TESTS DONE pass %d fail %d\n", pass, fail);
    std::fflush(stdout);
}
static bool search_is_null_move_local(const Move& m) {
    return m.from == 0 && m.to == 0 && m.promo == 0 && m.flags == 0;
}
static void search_extract_pv_from_tt(Position& root, int max_ply) {
    search_pv_table[0].length = 0;
    if (max_ply > 64) max_ply = 64;
    if (max_ply <= 0) return;
    Position tmp = root;
    Undo undo_stack[64];
    Move move_stack[64];
    int made = 0;
    for (int ply = 0; ply < max_ply; ++ply) {
        SearchTTEntry& entry = search_tt_entry(tmp.key);
        if (entry.key != tmp.key)
            break;
        Move mv = entry.move;
        if (search_is_null_move_local(mv))
            break;
        MoveList legal;
        generateLegalMoves(tmp, legal);
        bool found = false;
        for (int i = 0; i < legal.count; ++i) {
            if (move_equal(legal.m[i], mv)) {
                mv = legal.m[i];
                found = true;
                break;
            }
        }
        if (!found)
            break;
        search_pv_table[0].moves[ply] = mv;
        search_pv_table[0].length = ply + 1;
        makeMove(tmp, mv, undo_stack[made]);
        move_stack[made] = mv;
        ++made;
    }
    while (made > 0) {
        --made;
        unmakeMove(tmp, move_stack[made], undo_stack[made]);
    }
}
static void search_print_info(int depth, int score)
{
    long long elapsed = search_now() - search_limits.start_time;
    if (elapsed < 0) elapsed = 0;
    long long nps = elapsed > 0 ? (search_limits.nodes * 1000LL) / elapsed : search_limits.nodes;
    std::printf("info depth %d score cp %d nodes %lld nps %lld time %lld pv",
        depth, score, search_limits.nodes, nps, elapsed);
    for (int i = 0; i < search_pv_table[0].length; i++)
    {
        char buf[8];
        move_str(search_pv_table[0].moves[i], buf);
        std::printf(" %s", buf);
    }
    std::printf("\n");
    std::fflush(stdout);
}
// search() теперь принимает свою (по значению!) копию истории партии для
// детекции повторов. Внутри она свободно расширяется/сжимается push/pop-ами
// вдоль дерева поиска и просто уничтожается при выходе из функции — реальная
// история партии (владелец — UCI-слой) при этом не трогается.
Move search(Position& pos, int max_depth, int time_limit_ms, RepetitionStack rep = RepetitionStack{}, uint64_t node_limit = 0) {
    search_telemetry = SearchTelemetry{};
    search_limits.start_time = search_now();
    search_limits.time_limit = time_limit_ms;
    search_limits.check_time = (time_limit_ms > 0);
    search_limits.node_limit = node_limit;
    search_limits.check_nodes = (node_limit > 0);
    search_stop_reason = SEARCH_STOP_NONE;
    search_limits.stopped = false;
    search_limits.nodes = 0;
    g_search_initialized = true;
    std::memset(search_killer_moves, 0, sizeof(search_killer_moves));
    std::memset(search_history_moves, 0, sizeof(search_history_moves));
    for (SearchPVTable& pv : search_pv_table) pv = SearchPVTable{};
    Move best_move = { 0,0,0,0 };
    // build0.1: нормализация repetition-истории.
    // rep передан ПО ЗНАЧЕНИЮ (см. сигнатуру выше), поэтому мутация здесь
    // локальна и никогда не протекает обратно в вызывающий код. Это
    // гарантирует инвариант "rep.keys[rep.count-1] == pos.key" для ЛЮБОГО
    // пути, которым можно попасть в search(): прямые вызовы из bench/тестов
    // (пустой rep по умолчанию) и вызовы из UCI (uci_repetition уже содержит
    // root, здесь просто ничего лишнего не произойдёт).
    if (rep.count == 0 || rep.keys[rep.count - 1] != pos.key)
        rep.push(pos.key);
    // ========== ИСПРАВЛЕНИЕ: Проверка наличия легальных ходов ==========
    MoveList legal_check;
    generateLegalMoves(pos, legal_check);
    if (legal_check.count == 0) {
        // Игра окончена (мат или пат), возвращаем нулевой ход немедленно
        bool in_check = inCheck(pos, pos.sideToMove);
        search_set_reason_if_none(in_check ? SEARCH_STOP_MATE : SEARCH_STOP_DEPTH);
        std::printf("info string game over (%s)\n", in_check ? "checkmate" : "stalemate");
        std::fflush(stdout);
        return best_move;
    }
    // Fallback: если поиск остановится слишком рано по времени,
    // движок всё равно вернёт легальный ход, а не 0000.
    best_move = legal_check.m[0];
    // ===================================================================
    // v03b: clean minimal aspiration windows.
    // Old loop reused alpha/beta as score storage and could search early depths
    // with an accidental near-zero window, causing avoidable re-searches.
    // New policy:
    //   depth 1-4: full window, no aspiration
    //   depth 5+: previous score +/- 100 cp
    //   fail-low/high: one full-window re-search
    int last_score = 0;
    bool have_last_score = false;
    constexpr int ASPIRATION_START_DEPTH = 5;
    constexpr int ASPIRATION_WINDOW = 100;
    // build1.0: time management state. tm_soft is captured from uci_go's
    // handoff and the global is cleared immediately, so direct search()
    // callers (bench, tests, movetime/depth/nodes) never see it.
    const long long tm_soft = g_tm_soft_ms;
    g_tm_soft_ms = 0;
    int tm_factor_pct = 100;   // current soft-limit scale, percent
    int tm_stable = 0;         // consecutive iterations with same bestmove
    bool tm_have_prev = false;
    Move tm_prev_move = { 0,0,0,0 };
    int tm_prev_score = 0;
    for (int d = 1; d <= max_depth; ++d) {
        // build1.0: soft stop — never before depth 2 (we always have a
        // searched move), checked only between iterations. The hard limit
        // keeps guarding inside the iteration as in build0.9.
        if (tm_soft > 0 && d > 1) {
            long long elapsed = search_now() - search_limits.start_time;
            if (elapsed >= tm_soft * tm_factor_pct / 100) {
                search_telemetry.tmSoftStops++;
                search_set_reason_if_none(SEARCH_STOP_SOFT_TIME);
                break;
            }
        }
        int alpha = -SEARCH_INFINITE_SCORE;
        int beta = SEARCH_INFINITE_SCORE;
        bool used_aspiration = false;
        if (have_last_score && d >= ASPIRATION_START_DEPTH &&
            last_score > -SEARCH_MATE_SCORE + 200 &&
            last_score < SEARCH_MATE_SCORE - 200) {
            alpha = last_score - ASPIRATION_WINDOW;
            beta = last_score + ASPIRATION_WINDOW;
            used_aspiration = true;
        }
        int score = search_negamax(pos, alpha, beta, d, 0, rep);
        if (search_limits.stopped) break;
        if (used_aspiration && (score <= alpha || score >= beta)) {
            search_telemetry.aspirationResearches++;
            score = search_negamax(pos, -SEARCH_INFINITE_SCORE, SEARCH_INFINITE_SCORE, d, 0, rep);
            if (search_limits.stopped) break;
        }
        last_score = score;
        have_last_score = true;
        // FIX для Arena/PV: если дочерний узел вернулся из TT-cutoff,
        // обычная PV-таблица может содержать только первый ход.
        // Восстанавливаем линию по TT и берём более длинный вариант.
        SearchPVTable pv_from_search = search_pv_table[0];
        search_extract_pv_from_tt(pos, d);
        if (search_pv_table[0].length < pv_from_search.length)
            search_pv_table[0] = pv_from_search;
        if (search_pv_table[0].length == 0 && !search_is_null_move_local(best_move)) {
            search_pv_table[0].moves[0] = best_move;
            search_pv_table[0].length = 1;
        }
        if (search_pv_table[0].length > 0)
            best_move = search_pv_table[0].moves[0];
        // build1.0: bestmove-stability factor for the NEXT soft-limit check.
        // Same move as previous completed iteration -> shrink the budget
        // (1/2/3/4+ stable: 100/85/70/55%); move changed -> extend to 125%;
        // score dropped >40cp (fail-low) -> 135%. The hard limit still caps
        // everything, so extensions can never flag.
        if (tm_soft > 0) {
            bool fail_low = tm_have_prev && score <= tm_prev_score - 40;
            bool changed = tm_have_prev &&
                (tm_prev_move.from != best_move.from ||
                    tm_prev_move.to != best_move.to ||
                    tm_prev_move.promo != best_move.promo);
            if (tm_have_prev) tm_stable = changed ? 0 : tm_stable + 1;
            if (fail_low) {
                tm_factor_pct = 135;
                search_telemetry.tmFailLowExtends++;
            }
            else if (!tm_have_prev) tm_factor_pct = 100;
            else if (changed) tm_factor_pct = 125;
            else if (tm_stable >= 4) tm_factor_pct = 55;
            else if (tm_stable == 3) tm_factor_pct = 70;
            else if (tm_stable == 2) tm_factor_pct = 85;
            else tm_factor_pct = 100;
            tm_prev_move = best_move;
            tm_prev_score = score;
            tm_have_prev = true;
        }
        if (!search_limits.stopped)
        {
            search_print_info(d, score);
            if (score >= SEARCH_MATE_SCORE - 100 ||
                score <= -SEARCH_MATE_SCORE + 100) {
                search_set_reason_if_none(SEARCH_STOP_MATE);
                break;
            }
        }
    }
    search_set_reason_if_none(SEARCH_STOP_DEPTH);
    return best_move;
}
static bool position_equals(const Position& a, const Position& b) {
    if (a.sideToMove != b.sideToMove || a.castle != b.castle || a.ep != b.ep ||
        a.halfmove != b.halfmove || a.fullmove != b.fullmove || a.key != b.key ||
        a.ksq[WHITE] != b.ksq[WHITE] || a.ksq[BLACK] != b.ksq[BLACK]) return false;
    return std::memcmp(a.piece, b.piece, sizeof(a.piece)) == 0 &&
        std::memcmp(a.bbPieces, b.bbPieces, sizeof(a.bbPieces)) == 0 &&
        std::memcmp(a.bbOccupied, b.bbOccupied, sizeof(a.bbOccupied)) == 0 &&
        a.bbOccupiedAll == b.bbOccupiedAll;
}
static void run_evaluation_tests() {
    Position a, b, before;
    // build0.5: mirrored passed-pawn geometry regression cases.
    parse_fen(a, "4k3/8/4p3/8/4P3/8/8/4K3 w - - 0 1");
    check(!ev_is_passed_pawn(a, WHITE, sq_of("e4")), "passed white same file ahead");
    parse_fen(a, "4k3/8/3p4/8/4P3/8/8/4K3 w - - 0 1");
    check(!ev_is_passed_pawn(a, WHITE, sq_of("e4")), "passed white left file ahead");
    parse_fen(a, "4k3/8/5p2/8/4P3/8/8/4K3 w - - 0 1");
    check(!ev_is_passed_pawn(a, WHITE, sq_of("e4")), "passed white right file ahead");
    parse_fen(a, "4k3/8/8/8/4P3/3p4/8/4K3 w - - 0 1");
    check(ev_is_passed_pawn(a, WHITE, sq_of("e4")), "passed white enemy behind");
    parse_fen(a, "4k3/8/8/4p3/8/3P4/8/4K3 b - - 0 1");
    check(!ev_is_passed_pawn(a, BLACK, sq_of("e5")), "passed black enemy ahead");
    parse_fen(a, "4k3/8/3P4/4p3/8/8/8/4K3 b - - 0 1");
    check(ev_is_passed_pawn(a, BLACK, sq_of("e5")), "passed black enemy behind");
    set_start(a);
    before = a;
    check(evaluate(a) > -30 && evaluate(a) < 30, "eval start balanced");
    check(position_equals(a, before), "eval does not modify position");
    parse_fen(a, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    parse_fen(b, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    check(evaluate(a) == evaluate(b), "eval start side symmetry");
    parse_fen(a, "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    parse_fen(b, "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    check(evaluate(a) > evaluate(b), "eval extra pawn");
    parse_fen(a, "4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    parse_fen(b, "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    check(evaluate(a) > evaluate(b) + 350, "eval extra rook");
    parse_fen(a, "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    parse_fen(b, "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1");
    check(evaluate(a) > 0 && evaluate(b) < 0, "eval side relative");
    parse_fen(a, "4k3/8/4P3/8/8/8/8/4K3 w - - 0 1");
    parse_fen(b, "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    check(evaluate(a) > evaluate(b), "eval advanced pawn");
    parse_fen(a, "4k3/8/8/8/8/8/P7/R3K3 w - - 0 1");
    parse_fen(b, "4k3/8/8/8/8/8/P7/4K3 w - - 0 1");
    check(evaluate(a) > evaluate(b), "eval rook material");
}
static bool search_move_is_legal_and_roundtrips(Position& pos, const Move& mv) {
    Position copy = pos;
    Undo u;
    if (!search_make_move(pos, mv, u)) return false;
    search_unmake_move(pos, mv, u);
    return position_equals(pos, copy);
}
static void run_search_tests() {
    clear_search_tt();
    Position pos;
    set_start(pos);
    Position before = pos;
    Move m = search(pos, 4, 1000);
    if (m.from == 0 && m.to == 0) std::cout << "SEARCH Test 1: FAIL (no move)" << std::endl;
    else std::cout << "SEARCH Test 1: OK" << std::endl;
    if (position_equals(pos, before)) std::cout << "SEARCH Test 2: OK (start position restored)" << std::endl;
    else std::cout << "SEARCH Test 2: FAIL (position changed)" << std::endl;
    parse_fen(pos, "7k/6Q1/6K1/8/8/8/8/8 w - - 0 1");
    before = pos;
    m = search(pos, 3, 100);
    if (search_move_is_legal_and_roundtrips(pos, m)) std::cout << "SEARCH Test 3: OK" << std::endl;
    else std::cout << "SEARCH Test 3: FAIL" << std::endl;
    if (position_equals(pos, before)) std::cout << "SEARCH Test 4: OK (mate position restored)" << std::endl;
    else std::cout << "SEARCH Test 4: FAIL (mate position changed)" << std::endl;
    parse_fen(pos, "3k4/8/3K4/8/8/8/8/3R4 b - - 0 1");
    before = pos;
    m = search(pos, 3, 100);
    if (search_move_is_legal_and_roundtrips(pos, m)) std::cout << "SEARCH Test 5: OK" << std::endl;
    else std::cout << "SEARCH Test 5: FAIL" << std::endl;
    if (position_equals(pos, before)) std::cout << "SEARCH Test 6: OK (tactical position restored)" << std::endl;
    else std::cout << "SEARCH Test 6: FAIL (tactical position changed)" << std::endl;
}

static void run_repetition_tests() {
    int pass = 0;
    int fail = 0;
    auto result = [&](bool ok, const char* name) {
        if (ok) {
            ++pass;
            std::printf("info string REP %s OK\n", name);
        }
        else {
            ++fail;
            std::printf("info string REP %s FAIL\n", name);
        }
    };
    Position p, q;
    bool ok_p = parse_fen(p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    bool ok_q = parse_fen(q, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    result(ok_p && ok_q, "fen_setup");
    if (ok_p && ok_q) {
        RepetitionStack one;
        one.push(p.key);
        result(!is_repetition(p, one), "single_position_not_repeat");

        RepetitionStack prior;
        prior.push(p.key);
        prior.push(q.key);
        prior.push(p.key);
        Position repeated = p;
        repeated.halfmove = 2;
        result(is_repetition(repeated, prior), "same_key_two_plies_back");
        result(!is_repetition(q, prior), "different_key_not_repeat");

        Position after_irreversible = p;
        after_irreversible.halfmove = 0;
        result(!is_repetition(after_irreversible, prior), "irreversible_boundary_respected");
    }
    std::printf("info string REP TESTS DONE pass %d fail %d\n", pass, fail);
    std::fflush(stdout);
}

static void run_tt_tests() {
    int pass = 0;
    int fail = 0;
    auto result = [&](bool ok, const char* name) {
        if (ok) {
            ++pass;
            std::printf("info string TT %s OK\n", name);
        }
        else {
            ++fail;
            std::printf("info string TT %s FAIL\n", name);
        }
    };
    Position p;
    bool ok = parse_fen(p, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    RepetitionStack rep;
    if (ok) rep.push(p.key);
    clear_search_tt();
    Move first = ok ? search(p, 4, 0, rep, 0) : Move{0, 0, 0, 0};
    uint64_t first_hits = search_telemetry.ttHits;
    Move second = ok ? search(p, 4, 0, rep, 0) : Move{0, 0, 0, 0};
    uint64_t second_hits = search_telemetry.ttHits;
    result(ok, "fen_setup");
    result(ok && move_equal(first, second), "repeat_search_same_bestmove");
    result(ok && second_hits > 0, "tt_reused_on_repeat_search");
    std::printf("info string TT first_hits %llu second_hits %llu\n",
        (unsigned long long)first_hits, (unsigned long long)second_hits);
    std::printf("info string TT TESTS DONE pass %d fail %d\n", pass, fail);
    std::fflush(stdout);
}

static void run_final_audit() {
    std::printf("info string AUDIT start\n");
    run_bitboard_tests();

    g_pass = 0;
    g_fail = 0;
    run_evaluation_tests();
    std::printf("info string EVAL TESTS DONE pass %d fail %d\n", g_pass, g_fail);
    std::fflush(stdout);

    run_search_tests();
    run_repetition_tests();
    run_tt_tests();
    run_p0_tests();
    run_ep_tests();
    run_see_tests();
    run_perft_tests();
    std::printf("info string AUDIT done\n");
    std::fflush(stdout);
}
// ============================================================================
// UCI ADAPTER
// ============================================================================
static Position uci_pos;
// Реальная история партии для детекции повторов. Владелец — UCI-слой.
// search() получает КОПИЮ этого стека на каждый вызов.
static RepetitionStack uci_repetition;
static int uci_read_depth_after(const char* p) {
    while (*p == ' ') ++p;
    if (*p < '0' || *p > '9') return -1;
    return std::atoi(p);
}
static void uci_setoption(const char* line) {
    // Supported:
    //   setoption name Hash value 128
    //   setoption name Clear Hash
    // Arena and most GUIs send exactly this form for standard Hash.
    stop_search();
    std::string s(line);
    if (s.find("name Clear Hash") != std::string::npos) {
        clear_search_tt();
        unsigned long long actual_mb = (unsigned long long)((search_tt.size() * sizeof(SearchTTEntry)) / (1024ULL * 1024ULL));
        std::printf("info string hash cleared entries %llu actual %llu MB\n",
            (unsigned long long)search_tt.size(), actual_mb);
        std::fflush(stdout);
        return;
    }
    size_t name_pos = s.find("name Hash");
    if (name_pos != std::string::npos) {
        size_t value_pos = s.find("value", name_pos);
        if (value_pos == std::string::npos) return;
        int mb = std::atoi(s.c_str() + value_pos + 5);
        int clamped = search_clamp_hash_mb(mb);
        if (resize_search_tt(clamped)) {
            unsigned long long actual_mb = (unsigned long long)((search_tt.size() * sizeof(SearchTTEntry)) / (1024ULL * 1024ULL));
            std::printf("info string hash set %d MB requested, actual %llu MB entries %llu entry_size %llu\n",
                search_hash_mb,
                actual_mb,
                (unsigned long long)search_tt.size(),
                (unsigned long long)sizeof(SearchTTEntry));
        }
        else {
            std::printf("info string hash allocation failed, keeping %d MB entries %llu\n",
                search_hash_mb,
                (unsigned long long)search_tt.size());
        }
        std::fflush(stdout);
    }
}
static void uci_perft_current_position(int depth, bool divide) {
    stop_search();
    if (depth < 0) {
        std::printf("info string usage: perft <depth>, divide <depth>, or go perft <depth>\n");
        std::fflush(stdout);
        return;
    }
    long long start = search_now();
    uint64_t nodes = divide ? perft_divide(uci_pos, depth) : perft(uci_pos, depth);
    long long elapsed = search_now() - start;
    if (elapsed < 0) elapsed = 0;
    unsigned long long nps = elapsed > 0
        ? (unsigned long long)((nodes * 1000ULL) / (uint64_t)elapsed)
        : (unsigned long long)nodes;
    std::printf("info string perft depth %d nodes %llu time %lld nps %llu\n",
        depth,
        (unsigned long long)nodes,
        elapsed,
        nps);
    std::fflush(stdout);
}
// ============================================================================
// build0.4: telemetry printing helper.
//
// Used for both a single bench position and the aggregate across all bench
// positions. Percentages are always derived from whatever SearchTelemetry
// struct is passed in — for the aggregate call this means the percentages
// are computed from aggregate SUMS (not an average of per-position
// percentages), as required. Zero denominators are handled safely (0%).
// ============================================================================
static double telemetry_pct(uint64_t num, uint64_t den) {
    if (den == 0) return 0.0;
    return (double)num * 100.0 / (double)den;
}
static void print_telemetry_line(const char* label, const SearchTelemetry& t) {
    double ttHitRate = telemetry_pct(t.ttHits, t.ttProbes);
    double ttCutoffRate = telemetry_pct(t.ttCutoffs, t.ttHits);
    double firstMoveCutoffRate = telemetry_pct(t.firstMoveCutoffs, t.betaCutoffs);
    double nullCutoffRate = telemetry_pct(t.nullCutoffs, t.nullTries);
    double lmrResearchRate = telemetry_pct(t.lmrResearches, t.lmrReductions);
    std::printf(
        "info string telemetry %s qnodes %llu ttProbes %llu ttHits %llu ttHitRate %.2f%% "
        "ttCutoffs %llu ttCutoffRate %.2f%% betaCutoffs %llu firstMoveCutoffs %llu "
        "firstMoveCutoffRate %.2f%% nullTries %llu nullCutoffs %llu nullCutoffRate %.2f%% "
        "lmrReductions %llu lmrResearches %llu lmrResearchRate %.2f%% aspirationResearches %llu "
        "seeCalls %llu seePruned %llu futilityPruned %llu deltaPruned %llu "
        "tmSoftStops %llu tmFailLowExtends %llu\n",
        label,
        (unsigned long long)t.qnodes,
        (unsigned long long)t.ttProbes,
        (unsigned long long)t.ttHits,
        ttHitRate,
        (unsigned long long)t.ttCutoffs,
        ttCutoffRate,
        (unsigned long long)t.betaCutoffs,
        (unsigned long long)t.firstMoveCutoffs,
        firstMoveCutoffRate,
        (unsigned long long)t.nullTries,
        (unsigned long long)t.nullCutoffs,
        nullCutoffRate,
        (unsigned long long)t.lmrReductions,
        (unsigned long long)t.lmrResearches,
        lmrResearchRate,
        (unsigned long long)t.aspirationResearches,
        (unsigned long long)t.seeCalls,
        (unsigned long long)t.seePruned,
        (unsigned long long)t.futilityPruned,
        (unsigned long long)t.deltaPruned,
        (unsigned long long)t.tmSoftStops,
        (unsigned long long)t.tmFailLowExtends);
    std::fflush(stdout);
    if (!(t.ttCutoffs <= t.ttHits && t.ttHits <= t.ttProbes)) {
        std::printf(
            "info string telemetry warning %s invariant violated: ttCutoffs<=ttHits<=ttProbes (%llu/%llu/%llu)\n",
            label,
            (unsigned long long)t.ttCutoffs,
            (unsigned long long)t.ttHits,
            (unsigned long long)t.ttProbes);
        std::fflush(stdout);
    }
    if (!(t.firstMoveCutoffs <= t.betaCutoffs)) {
        std::printf(
            "info string telemetry warning %s invariant violated: firstMoveCutoffs<=betaCutoffs (%llu/%llu)\n",
            label,
            (unsigned long long)t.firstMoveCutoffs,
            (unsigned long long)t.betaCutoffs);
        std::fflush(stdout);
    }
    if (!(t.nullCutoffs <= t.nullTries)) {
        std::printf(
            "info string telemetry warning %s invariant violated: nullCutoffs<=nullTries (%llu/%llu)\n",
            label,
            (unsigned long long)t.nullCutoffs,
            (unsigned long long)t.nullTries);
        std::fflush(stdout);
    }
    if (!(t.lmrResearches <= t.lmrReductions)) {
        std::printf(
            "info string telemetry warning %s invariant violated: lmrResearches<=lmrReductions (%llu/%llu)\n",
            label,
            (unsigned long long)t.lmrResearches,
            (unsigned long long)t.lmrReductions);
        std::fflush(stdout);
    }
}
static void uci_bench(int depth) {
    stop_search();
    if (depth <= 0)
        depth = 6;
    struct BenchPosition {
        const char* name;
        const char* fen;
    };
    static const BenchPosition bench_positions[] = {
        {
            "startpos",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        },
        {
            "kiwipete",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        },
        {
            "castles",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        },
        {
            "rook_endgame",
            "3k4/8/3K4/8/8/8/8/3R4 b - - 0 1"
        }
    };
    clear_search_tt();
    long long total_nodes = 0;
    long long total_start = search_now();
    SearchTelemetry aggregate{};
    std::printf("info string bench start depth %d positions %d\n",
        depth,
        (int)(sizeof(bench_positions) / sizeof(bench_positions[0])));
    std::fflush(stdout);
    for (const BenchPosition& bp : bench_positions) {
        Position p;
        if (!parse_fen(p, bp.fen)) {
            std::printf("info string bench %s bad fen\n", bp.name);
            continue;
        }
        long long start = search_now();
        Move best = search(p, depth, 0);
        long long elapsed = search_now() - start;
        if (elapsed < 0) elapsed = 0;
        long long nodes = search_limits.nodes;
        total_nodes += nodes;
        long long nps = elapsed > 0 ? (nodes * 1000LL) / elapsed : nodes;
        char best_buf[8];
        if (!search_is_null_move_local(best)) {
            move_str(best, best_buf);
        }
        else {
            // Не используем strcpy: MSVC ругается C4996.
            best_buf[0] = '0';
            best_buf[1] = '0';
            best_buf[2] = '0';
            best_buf[3] = '0';
            best_buf[4] = 0;
        }
        std::printf("info string bench %s depth %d bestmove %s nodes %lld time %lld nps %lld\n",
            bp.name,
            depth,
            best_buf,
            nodes,
            elapsed,
            nps);
        std::fflush(stdout);
        // build0.4: search() is fully finished by this point (called
        // synchronously, not via g_search_thread), so this copy+print never
        // races with an active search and never happens from a UCI thread
        // while search is running.
        SearchTelemetry t = search_telemetry;
        print_telemetry_line(bp.name, t);
        aggregate.qnodes += t.qnodes;
        aggregate.ttProbes += t.ttProbes;
        aggregate.ttHits += t.ttHits;
        aggregate.ttCutoffs += t.ttCutoffs;
        aggregate.betaCutoffs += t.betaCutoffs;
        aggregate.firstMoveCutoffs += t.firstMoveCutoffs;
        aggregate.nullTries += t.nullTries;
        aggregate.nullCutoffs += t.nullCutoffs;
        aggregate.lmrReductions += t.lmrReductions;
        aggregate.lmrResearches += t.lmrResearches;
        aggregate.aspirationResearches += t.aspirationResearches;
        aggregate.seeCalls += t.seeCalls;
        aggregate.seePruned += t.seePruned;
        aggregate.futilityPruned += t.futilityPruned;
        aggregate.deltaPruned += t.deltaPruned;
        aggregate.tmSoftStops += t.tmSoftStops;
        aggregate.tmFailLowExtends += t.tmFailLowExtends;
    }
    long long total_elapsed = search_now() - total_start;
    if (total_elapsed < 0) total_elapsed = 0;
    long long total_nps = total_elapsed > 0 ? (total_nodes * 1000LL) / total_elapsed : total_nodes;
    std::printf("info string bench done depth %d total_nodes %lld total_time %lld total_nps %lld\n",
        depth,
        total_nodes,
        total_elapsed,
        total_nps);
    std::fflush(stdout);
    print_telemetry_line("aggregate", aggregate);
}
static bool get_legal_mv(const Position& pos, const char* s, Move& out) {
    MoveList ml;
    generateLegalMoves(pos, ml);
    char buf[8];
    for (int i = 0; i < ml.count; ++i) {
        move_str(ml.m[i], buf);
        if (!std::strcmp(buf, s)) {
            out = ml.m[i];
            return true;
        }
    }
    return false;
}
// build0.2 FIX 3: убран фиксированный лимит истории партии в 256 полуходов.
// makeMove() применяется напрямую (Undo нам тут больше не нужен за пределами
// самого вызова), а repetition stack сбрасывается на каждом необратимом ходе
// (halfmove == 0 — пешечный ход или взятие), т.к. более старые позиции уже
// не могут участвовать в повторении, и это же не даёт RepetitionStack
// переполниться на длинной партии.
static void uci_apply_move(Position& pos, const char* s) {
    Move m{};
    if (!get_legal_mv(pos, s, m)) {
        std::printf("info string illegal move %s\n", s);
        return;
    }
    Undo scratch;
    makeMove(pos, m, scratch);
    if (pos.halfmove == 0)
        uci_repetition.count = 0;
    uci_repetition.push(pos.key);
}
static void uci_position(const char* line) {
    stop_search();
    uci_repetition.count = 0;
    // set_start() здесь — только предварительный/дефолтный сброс доски.
    // Его ключ намеренно НЕ пушится в uci_repetition сейчас — корнем
    // repetition-истории должна быть ОКОНЧАТЕЛЬНАЯ позиция (startpos или
    // успешно разобранный FEN), а не этот промежуточный placeholder.
    set_start(uci_pos);
    const char* p = line + 8;
    while (*p == ' ') ++p;
    if (std::strncmp(p, "startpos", 8) == 0) {
        p += 8;
    }
    else if (std::strncmp(p, "fen", 3) == 0) {
        p += 3;
        while (*p == ' ') ++p;
        char fen[256];
        int i = 0;
        while (*p && std::strncmp(p, "moves", 5) != 0 && i < 255) fen[i++] = *p++;
        while (i > 0 && fen[i - 1] == ' ') --i;
        fen[i] = 0;
        if (!parse_fen(uci_pos, fen)) {
            std::printf("info string bad fen\n");
            // build0.1: немедленно выходим при некорректном FEN — не
            // добавляем повреждённую/неопределённую позицию в repetition
            // stack и не пытаемся применять ходы поверх неё.
            return;
        }
    }
    else return;
    // build0.1: root-позиция теперь окончательно определена (startpos или
    // валидный FEN). Пушим её ровно один раз, до применения любых ходов —
    // это гарантирует, что "position startpos" / "position fen X" без
    // списка moves всё равно удовлетворяет
    // rep.keys[rep.count-1] == uci_pos.key.
    uci_repetition.push(uci_pos.key);
    while (*p == ' ') ++p;
    if (std::strncmp(p, "moves", 5) == 0) {
        p += 5;
        while (*p) {
            while (*p == ' ') ++p;
            if (!*p) break;
            char mv[8] = {};
            int j = 0;
            while (*p && *p != ' ' && j < 7) mv[j++] = *p++;
            mv[j] = 0;
            uci_apply_move(uci_pos, mv);
        }
    }
}
static bool is_null_move(const Move& m)
{
    return m.from == 0 &&
        m.to == 0 &&
        m.flags == 0;
}
static void uci_go(const char* line) {
    // build0.6: join any previous search before reading uci_pos. The old
    // order read sideToMove while the search thread could be making moves.
    stop_search();
    int depth = 0, movetime = 0;
    int wtime = 0, btime = 0, winc = 0, binc = 0;
    uint64_t node_limit = 0;
    bool nodes_specified = false;
    const char* p = line + 2;
    while (*p) {
        while (*p == ' ') ++p;
        if (std::strncmp(p, "depth", 5) == 0) {
            p += 5; while (*p == ' ') ++p;
            depth = std::atoi(p);
        }
        else if (std::strncmp(p, "movetime", 8) == 0) {
            p += 8; while (*p == ' ') ++p;
            movetime = std::atoi(p);
        }
        else if (std::strncmp(p, "wtime", 5) == 0) {
            p += 5; while (*p == ' ') ++p;
            wtime = std::atoi(p);
        }
        else if (std::strncmp(p, "btime", 5) == 0) {
            p += 5; while (*p == ' ') ++p;
            btime = std::atoi(p);
        }
        else if (std::strncmp(p, "winc", 4) == 0) {
            p += 4; while (*p == ' ') ++p;
            winc = std::atoi(p);
        }
        else if (std::strncmp(p, "binc", 4) == 0) {
            p += 4; while (*p == ' ') ++p;
            binc = std::atoi(p);
        }
        else if (std::strncmp(p, "nodes", 5) == 0 &&
            (p[5] == ' ' || p[5] == 0)) {
            p += 5; while (*p == ' ') ++p;
            // A present but malformed/negative/overflowing nodes token is
            // safely treated as a zero budget, never as an unlimited search.
            nodes_specified = true;
            node_limit = 0;
            if (*p >= '0' && *p <= '9') {
                errno = 0;
                char* endptr = nullptr;
                unsigned long long parsed = std::strtoull(p, &endptr, 10);
                if (errno != ERANGE && endptr != p &&
                    (*endptr == 0 || *endptr == ' ')) {
                    const unsigned long long maximum =
                        (unsigned long long)LLONG_MAX;
                    if (parsed > maximum) parsed = maximum;
                    node_limit = (uint64_t)parsed;
                    p = endptr;
                }
            }
        }
        while (*p && *p != ' ') ++p;
    }
    // build1.0: soft/hard time budget replaces the flat "available/30 + inc".
    // Soft (don't start a new iteration past it) = available/25 + 0.7*inc.
    // Hard (abort switch in the node loop, as before) = min(4*soft,
    // available/4, available - 50ms lag buffer). The increment is weighted
    // 0.7 and capped by the remaining clock, so the engine can never bet
    // time it has not yet earned (the old formula could flag on
    // "remaining 100ms + inc 1000ms"). Explicit "go movetime" bypasses all
    // of this: g_tm_soft_ms stays 0 and behavior is identical to build0.9.
    g_tm_soft_ms = 0;
    if (movetime <= 0) {
        int available = (uci_pos.sideToMove == WHITE) ? wtime : btime;
        if (available > 0 || winc > 0 || binc > 0) {
            long long inc = (uci_pos.sideToMove == WHITE) ? winc : binc;
            long long soft = (long long)available / 25 + inc * 7 / 10;
            long long hard = soft * 4;
            if (hard > (long long)available / 4) hard = (long long)available / 4;
            long long cap = (long long)available - 50;
            if (cap < 1) cap = 1;
            if (hard > cap) hard = cap;
            if (hard < 5) hard = 5;
            if (soft > hard) soft = hard;
            if (soft < 1) soft = 1;
            movetime = (int)hard;
            g_tm_soft_ms = soft;
        }
    }
    if (depth <= 0) depth = 64;

    // build0.6: an explicit zero (or negative) budget starts no search but
    // still returns one legal fallback move as required by UCI game play.
    if (nodes_specified && node_limit == 0) {
        MoveList legal;
        generateLegalMoves(uci_pos, legal);
        if (legal.count == 0) {
            std::printf("info string node budget requested 0 searched 0 reason nodes\n");
            std::printf("bestmove 0000\n");
        }
        else {
            char buf[8];
            move_str(legal.m[0], buf);
            std::printf("info string node budget requested 0 searched 0 reason nodes\n");
            std::printf("bestmove %s\n", buf);
        }
        std::fflush(stdout);
        return;
    }

    search_limits.stopped = false;
    int depth_copy = depth;
    int movetime_copy = movetime;
    uint64_t node_limit_copy = nodes_specified ? node_limit : 0;
    g_search_initialized = false;
    g_search_running = true;
    g_search_thread = std::thread([depth_copy, movetime_copy, node_limit_copy]() {
        Move best = search(uci_pos, depth_copy, movetime_copy,
            uci_repetition, node_limit_copy);
        if (node_limit_copy > 0) {
            std::printf("info string node budget requested %llu searched %lld reason %s\n",
                (unsigned long long)node_limit_copy,
                search_limits.nodes,
                search_stop_reason_name());
        }
        char buf[8];
        if (!is_null_move(best)) {
            move_str(best, buf);
            std::printf("bestmove %s\n", buf);
        }
        else {
            std::printf("bestmove 0000\n");
        }
        std::fflush(stdout);
        g_search_running = false;
        });
    // Do not let the UCI loop consume a following "stop" until search() has
    // completed its one-time initialization, including stopped=false.
    while (!g_search_initialized.load() && g_search_thread.joinable())
        std::this_thread::yield();
}
int uci_main() {
    set_start(uci_pos);
    uci_repetition.count = 0;
    uci_repetition.push(uci_pos.key);
    char line[4096];
    while (std::fgets(line, sizeof(line), stdin)) {
        line[std::strcspn(line, "\r\n")] = 0;
        if (!line[0]) continue;
        if (!std::strcmp(line, "uci")) {
            std::printf("id name MyChessEngine build1.1-bitboard-stage11-magic\n");
            std::printf("id author AI Battle Project\n");
            std::printf("option name Hash type spin default %d min %d max %d\n",
                SEARCH_DEFAULT_HASH_MB, SEARCH_MIN_HASH_MB, SEARCH_MAX_HASH_MB);
            std::printf("option name Clear Hash type button\n");
            std::printf("uciok\n");
        }
        else if (!std::strcmp(line, "isready")) {
            std::printf("readyok\n");
        }
        else if (std::strncmp(line, "setoption ", 10) == 0) {
            uci_setoption(line);
        }
        else if (!std::strcmp(line, "ucinewgame")) {
            stop_search();
            clear_search_tt();
            std::memset(search_killer_moves, 0, sizeof(search_killer_moves));
            std::memset(search_history_moves, 0, sizeof(search_history_moves));
            uci_repetition.count = 0;
            set_start(uci_pos);
            uci_repetition.push(uci_pos.key);
        }
        else if (!std::strcmp(line, "d")) {
            stop_search();
            std::printf("info string side %s\ninfo string hash %llu\n",
                uci_pos.sideToMove == WHITE ? "white" : "black",
                (unsigned long long)uci_pos.key);
        }
        else if (!std::strcmp(line, "debug on")) { std::printf("info string debug enabled\n"); }
        else if (!std::strcmp(line, "debug off")) { std::printf("info string debug disabled\n"); }
        else if (!std::strcmp(line, "eval")) {
            stop_search();
            std::printf("info string eval cp %d\n", evaluate(uci_pos));
            std::fflush(stdout);
        }
        else if (!std::strcmp(line, "audit")) {
            stop_search();
            clear_search_tt();
            run_final_audit();
        }
        else if (!std::strcmp(line, "p0test")) {
            stop_search();
            clear_search_tt();
            run_p0_tests();
        }
        else if (!std::strcmp(line, "eptest")) {
            stop_search();
            run_ep_tests();
        }
        else if (!std::strcmp(line, "seetest")) {
            stop_search();
            run_see_tests();
        }
        else if (!std::strcmp(line, "perfttest")) {
            stop_search();
            run_perft_tests();
        }
        else if (!std::strcmp(line, "bbtest")) {
            stop_search();
            run_bitboard_tests();
        }
        else if (std::strncmp(line, "bench", 5) == 0 && (line[5] == ' ' || line[5] == 0)) {
            uci_bench(uci_read_depth_after(line + 5));
        }
        else if (std::strncmp(line, "perft", 5) == 0 && (line[5] == ' ' || line[5] == 0)) {
            uci_perft_current_position(uci_read_depth_after(line + 5), false);
        }
        else if (std::strncmp(line, "divide", 6) == 0 && (line[6] == ' ' || line[6] == 0)) {
            uci_perft_current_position(uci_read_depth_after(line + 6), true);
        }
        else if (std::strncmp(line, "position ", 9) == 0) {
            uci_position(line);
        }
        else if (std::strncmp(line, "go perft", 8) == 0 && (line[8] == ' ' || line[8] == 0)) {
            uci_perft_current_position(uci_read_depth_after(line + 8), false);
        }
        else if (!std::strncmp(line, "go", 2) && (line[2] == ' ' || line[2] == 0)) {
            uci_go(line);
        }
        else if (!std::strcmp(line, "stop")) {
            stop_search();
        }
        else if (!std::strcmp(line, "quit")) {
            stop_search();
            break;
        }
        std::fflush(stdout);
    }
    stop_search();
    return 0;
}
int main() {
    initZobrist();
    bb_init_attack_tables();
    // Build the read-only magic tables before the UCI loop so table
    // construction is not charged to the first search benchmark.
    bb_init_magic_tables();
    resize_search_tt(SEARCH_DEFAULT_HASH_MB);
    return uci_main();
}
