-- Independent synthetic rewrites; original.py is the correctness reference.
CREATE MACRO sum_squares(x) AS SUM(CAST(x AS DOUBLE) * CAST(x AS DOUBLE));
CREATE MACRO weighted_mean(x, w) AS
    CASE WHEN SUM(CASE WHEN x IS NULL OR w IS NULL THEN NULL::DOUBLE
                       ELSE CAST(w AS DOUBLE) END) = 0::DOUBLE
         THEN NULL::DOUBLE
         ELSE SUM(CAST(x AS DOUBLE) * CAST(w AS DOUBLE)) /
              SUM(CASE WHEN x IS NULL OR w IS NULL THEN NULL::DOUBLE
                       ELSE CAST(w AS DOUBLE) END)
    END;
