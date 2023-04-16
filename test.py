def decode_arch(arch):
        s = s.replace("[", "").replace("]", "").replace(" ", "").strip()
        rows = s.split('\n')
        result = []
        current_matrix = []
        for row in rows:
            if not row:
                result.append(current_matrix)
                current_matrix = []
            else:
                current_matrix.append([float(n) for n in row.split(",") if n])
        if current_matrix:
            result.append(current_matrix)
        return result
