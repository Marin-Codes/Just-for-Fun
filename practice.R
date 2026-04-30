a<- c(1,2,3,4,5)
b<-c(2,4,6,8,10)
cat(a,b)

#Objects

#  Vectors
numbers<-c(1,2,3,4,5,6)

#  Matrices
m<-matrix(1:12,nrow=4,ncol=3)
print(m)

#  Dataframe
df<-data.frame(
  id=1:4,
  name=c("Ava","Anne","Justin","Katy"),
  age=c(28,32,40,31)
)
print(df)

#  Lists
my_list<-list(
  name="Enhypen",
  album=c("Sin:the vanish","BloodMoon","Romance:Untold"),
  active=TRUE
)
print(my_list)


#  FUNCTIONS

#Built-in

num<-1:10
# some examples for built-in func.
mean(num)
sum(num)
sqrt(num)

# A special one (The sample() func.) it is basically random func. in R

sample(num,5) # it is simple random sampling (unique values)

sample(num,5,replace=TRUE) #it allows repetetion 



# Conditional Statement 

#   IF

 














