USE master;  
GO  
IF DB_ID (N'article_bias') IS NOT NULL  
DROP DATABASE article_bias;  
GO  
CREATE DATABASE article_bias
--Ensure that the database has a case sensitive collation for the ngram storage
COLLATE Latin1_General_CS_AS;  
GO  
--Verifying collation and option settings.  
SELECT name, collation_name 
FROM sys.databases  
WHERE name = N'article_bias';  
GO