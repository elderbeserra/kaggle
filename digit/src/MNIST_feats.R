get.features = function(x, block.intens = 2, block.zero = 2){
 
  get.blocks = function(x, size, FUN, sep = '-') {
   
    dummy = matrix(0, ncol = 28, nrow = 28)
    area = matrix(paste(ceiling(col(dummy)/size), ceiling(row(dummy)/size), sep = sep), nc = ncol(dummy))
   
    t(apply(x, 1, function(img){
      img = matrix(img, ncol = 28, nrow = 28, byrow = TRUE)
      tapply(img, area, FUN = FUN)
    }))
  }
 
  cat('Extraindo intensidade dos pixels...\n')
  intensities = get.blocks(x, block.intens, mean)
  colnames(intensities) <- paste0('px_intens_', 1:ncol(intensities))
  cat('Extraindo contagem dos zeros...\n')
  zeros = get.blocks(x, block.zero, FUN = function(r)sum(r == 0))
  colnames(zeros) <- paste0('zero_cnt_', 1:ncol(zeros))
  cat('Num. Features extraidas:', ncol(intensities) + ncol(zeros), '\n')
  cbind(intensities, zeros)
}

extrai.features = function(train.path, test.path){
  
  train = read.csv(train.path)
  Label = train$label
  train = as.matrix(subset(train, select = -label))
  
  test = as.matrix(read.csv(test.path))
  
  train = get.features(train)
  test = get.features(test)
  
  write.csv(train, 'train_px22_cnt_zero22.csv', quote = FALSE, row.names = FALSE)
  write.csv(test, 'test_px22_cnt_zero22.csv', quote = FALSE, row.names = FALSE)
  write.csv(Label, 'labels.csv', quote = FALSE, row.names = FALSE)
}


# extrai.features(train.path, test.path)