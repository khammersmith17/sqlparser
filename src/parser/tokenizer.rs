fn is_keyword(k: &str) -> bool {
    matches!(
        k,
        "PRIMARY"
            | "KEY"
            | "NOT"
            | "NULL"
            | "UNIQUE"
            | "DEFAULT"
            | "CHECK"
            | "COLLATE"
            | "AUTOINCREMENT"
            | "REFERENCES"
            | "ON"
            | "DELETE"
            | "UPDATE"
            | "DEFERRABLE"
            | "INITIALLY"
            | "DEFERRED"
            | "IMMEDIATE"
            | "ASC"
            | "DESC"
    )
}

#[derive(Debug)]
pub enum SQLiteKeyword {
    Primary,
    Key,
    Not,
    Null,
    Unique,
    Default,
    Check,
    Collate,
    AutoIncrement,
    References,
    On,
    Delete,
    Update,
    Deferrable,
    Initially,
    Deferred,
    Immediate,
    Ascending,
    Descending,
}

impl TryFrom<String> for SQLiteKeyword {
    type Error = String;
    fn try_from(value: String) -> Result<SQLiteKeyword, Self::Error> {
        match value.to_uppercase().as_str() {
            "PRIMARY" => Ok(Self::Primary),
            "KEY" => Ok(Self::Key),
            "NOT" => Ok(Self::Not),
            "NULL" => Ok(Self::Null),
            "UNIQUE" => Ok(Self::Unique),
            "DEFAULT" => Ok(Self::Default),
            "CHECK" => Ok(Self::Check),
            "COLLATE" => Ok(Self::Collate),
            "AUTOINCREMENT" => Ok(Self::AutoIncrement),
            "REFERENCES" => Ok(Self::References),
            "ON" => Ok(Self::On),
            "DELETE" => Ok(Self::Delete),
            "UPDATE" => Ok(Self::Update),
            "DEFERRABLE" => Ok(Self::Deferrable),
            "INITIALLY" => Ok(Self::Initially),
            "DEFERRED" => Ok(Self::Deferred),
            "IMMEDIATE" => Ok(Self::Immediate),
            "ASC" => Ok(Self::Ascending),
            "DESC" => Ok(Self::Descending),
            _ => Err("Invalid keyword".into()),
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum SqlConditionToken {
    Keyword(SQLiteKeyword),
    Identifier(String),
    QuotedIdentifier(String), // wrapped in ''
    StringGroup(String),      // wrapped in ""
    NumberGroup(String),
    ParenGroup(String),
    Symbol(String),
}

enum QuoteType {
    Single,
    Double,
}

struct Tokenizer<'a> {
    token: &'a [u8],
    cursor: usize,
    len: usize,
}

impl<'a> Tokenizer<'a> {
    fn new(input: &'a str) -> Tokenizer<'a> {
        let token = input.as_bytes();
        let cursor = 0_usize;
        let len = token.len();
        Tokenizer { token, cursor, len }
    }

    fn tokenize(mut self) -> Vec<SqlConditionToken> {
        let mut result: Vec<SqlConditionToken> = Vec::new();
        while self.skip_whitespace() {
            let Some(token_start) = self.peek_char() else {
                break;
            };
            let token = match token_start {
                '"' => self.parse_inner_quote(QuoteType::Double),
                '\'' => self.parse_inner_quote(QuoteType::Single),
                '(' => self.parse_inner_paren(),
                '0'..='9' | '+' | '-' => self.parse_number(),
                _ if token_start.is_alphanumeric() => self.parse_identifier(),
                _ => SqlConditionToken::Symbol(token_start.into()),
            };
            result.push(token);
        }
        result
    }

    fn peek_char(&self) -> Option<char> {
        if self.cursor < self.len {
            Some(self.token[self.cursor].into())
        } else {
            None
        }
    }

    fn bump_cursor(&mut self) -> Option<char> {
        if let Some(c) = self.peek_char() {
            self.cursor += 1;
            Some(c)
        } else {
            return None;
        }
    }

    fn skip_whitespace(&mut self) -> bool {
        while self.cursor < self.len
            && (self.token[self.cursor] == 32 || self.token[self.cursor] == 10)
        {
            self.cursor += 1
        }

        self.cursor < self.len
    }

    fn parse_inner_paren(&mut self) -> SqlConditionToken {
        // move passed (
        let _ = self.bump_cursor();
        let mut str = String::new();
        while let Some(c) = self.peek_char() {
            if c == ')' {
                break;
            }
            str.push(c);
            let _ = self.bump_cursor();
        }

        let _ = self.bump_cursor();
        SqlConditionToken::ParenGroup(str)
    }

    fn parse_inner_quote(&mut self, q_type: QuoteType) -> SqlConditionToken {
        // move passed quote
        let mut str = String::new();
        let sep = match q_type {
            QuoteType::Single => '\'',
            QuoteType::Double => '"',
        };
        let _ = self.bump_cursor();
        while let Some(c) = self.peek_char() {
            if c == sep {
                break;
            }
            str.push(c);
            let _ = self.bump_cursor();
        }

        match q_type {
            QuoteType::Single => SqlConditionToken::StringGroup(str),
            QuoteType::Double => SqlConditionToken::QuotedIdentifier(str),
        }
    }

    fn parse_number(&mut self) -> SqlConditionToken {
        let mut num_group = String::new();
        while let Some(c) = self.peek_char() {
            if c == ' ' {
                break;
            }
            num_group.push(c);
            let _ = self.bump_cursor();
        }
        SqlConditionToken::NumberGroup(num_group)
    }

    fn parse_identifier(&mut self) -> SqlConditionToken {
        let mut str = String::new();
        while let Some(c) = self.peek_char() {
            if !(c.is_alphanumeric() || c == '_') {
                break;
            }
            str.push(c.to_ascii_uppercase());
            let _ = self.bump_cursor();
        }

        if is_keyword(&str) {
            let kywrd = SQLiteKeyword::try_from(str)
                .expect("Invalid key word made it through the condition");
            SqlConditionToken::Keyword(kywrd)
        } else {
            SqlConditionToken::Identifier(str)
        }
    }
}

#[derive(Debug)]
pub struct TokenStream {
    inner: Vec<SqlConditionToken>,
    i: usize,
    len: usize,
}

impl TokenStream {
    pub fn new(input: &str) -> TokenStream {
        let tokenizer = Tokenizer::new(input);
        let inner = tokenizer.tokenize();
        let len = inner.len();
        TokenStream {
            inner,
            i: 0_usize,
            len,
        }
    }

    #[allow(dead_code)]
    pub fn tokens(&self) -> &[SqlConditionToken] {
        &self.inner
    }

    pub fn peek(&self) -> Option<&SqlConditionToken> {
        if self.i < self.len {
            Some(&self.inner[self.i])
        } else {
            None
        }
    }

    pub fn next(&mut self) -> Option<&SqlConditionToken> {
        if self.i < self.len {
            let item = &self.inner[self.i];
            self.i += 1;
            Some(&item)
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[allow(dead_code)]
    pub fn iter(&self) -> std::slice::Iter<SqlConditionToken> {
        self.inner.iter()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_primary_key() {
        let options = "primary key autoincrement";
        let token_stream = TokenStream::new(&options);
        println!("token stream: {:?}", token_stream);
    }

    #[test]
    fn test_not_null() {
        let options = "not null";
        let token_stream = TokenStream::new(&options);
        println!("token stream: {:?}", token_stream);

        for t in token_stream.iter() {
            println!("token: {:?}", t);
        }
    }
}
