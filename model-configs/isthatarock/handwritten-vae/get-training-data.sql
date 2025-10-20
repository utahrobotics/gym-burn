SELECT 
    train.row_id, 
    i1.webp as webp_input, 
    i2.webp as webp_expected, 
    i1.width as input_width, 
    i1.height as input_height, 
    i2.width as expected_width, 
    i2.height as expected_height 
FROM train
INNER JOIN images i1 ON train.input = i1.row_id
INNER JOIN images i2 ON train.expected = i2.row_id
WHERE train.row_id BETWEEN ?1 AND (?1 + ?2 - 1)