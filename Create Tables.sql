USE article_bias;  
GO

CREATE TABLE article (
	article_id		INT				NOT NULL IDENTITY(1,1),
	source_name		CHAR(6)			NOT NULL,
	/*2083 is the maximum length of a URL allowed by IE, therefore will use that as the max length
	  though it is unlikely that any of these URLs will approach that length
	  also using nvarchar instead of varchar to not worry about URL encoding */
	article_url		NVARCHAR(2083)	NOT NULL,
	raw_text		TEXT			NOT NULL,
	processed_text	TEXT			NULL,
	is_training		BIT				NULL,
	CONSTRAINT		pk_article		PRIMARY KEY(article_id)
);

CREATE TABLE ngram (
	ngram_id		INT				NOT NULL IDENTITY(1,1),
	/*Most words are well under 30 characters; this just ensures no words are cut off*/
	ngram			VARCHAR(30)		NOT NULL, 
	article_count	INT				NOT NULL,
	inv_doc_freq	FLOAT			NOT NULL,
	CONSTRAINT		pk_ngram		PRIMARY KEY NONCLUSTERED (ngram_id)
);

/*Creating a clustered index on NGram since most of the queries will be on this column
and not the primary key*/
CREATE UNIQUE CLUSTERED INDEX ix_ngram ON ngram (ngram);

CREATE TABLE article_ngram (
	article_id	INT					NOT NULL,
	ngram_id	INT					NOT NULL,
	term_freq	FLOAT				NOT NULL,
	tf_idf		FLOAT				NULL,
	CONSTRAINT	fk_article			FOREIGN KEY(article_id)				REFERENCES article(article_id),
	CONSTRAINT	fk_ngram			FOREIGN KEY(ngram_id)				REFERENCES ngram(ngram_id),
	CONSTRAINT	pk_article_ngram	PRIMARY KEY(article_id, ngram_id)
);