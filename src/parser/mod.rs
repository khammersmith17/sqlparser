mod record;
use record::Record;
use record::Value;
mod schema;
use anyhow::{Result, bail};
use schema::{IndexColumn, PrimaryKey, SQLiteColumnConstraints, SQLiteColumnType, TableColumn};
use std::rc::Rc;
mod tokenizer;
use tokenizer::{SQLiteKeyword, SqlConditionToken, TokenStream};

pub type FilterFn = Rc<dyn Fn(&Record) -> bool>;

// utility to parse commas as seperators
// accounting for the fact that there may be a comma in the
// middle of a group
struct GroupDepth {
    paren_depth: usize,
    single_qoute_depth: usize,
    double_qoute_depth: usize,
    bracket_depth: usize,
    depth: usize,
}

impl GroupDepth {
    fn new() -> GroupDepth {
        GroupDepth {
            paren_depth: 0_usize,
            single_qoute_depth: 0_usize,
            double_qoute_depth: 0_usize,
            bracket_depth: 0_usize,
            depth: 0_usize,
        }
    }

    fn update(&mut self, token: char) {
        // returns the max depth
        match token {
            '(' => self.paren_depth += 1,
            '[' => self.bracket_depth += 1,
            ']' => self.bracket_depth -= 1,
            ')' => self.paren_depth += 1,
            '\'' => {
                if self.single_qoute_depth % 2 == 1 {
                    self.single_qoute_depth -= 1
                } else {
                    self.single_qoute_depth += 1
                }
            }
            '"' => {
                if self.double_qoute_depth % 2 == 1 {
                    self.double_qoute_depth -= 1
                } else {
                    self.double_qoute_depth += 1
                }
            }
            _ => {}
        }

        self.depth = usize::max(
            self.paren_depth,
            usize::max(
                self.bracket_depth,
                usize::max(self.single_qoute_depth, self.double_qoute_depth),
            ),
        )
    }
}

pub fn generate_condition_evaluator(condition: Option<Condition>) -> FilterFn {
    if let Some(cond) = condition {
        // closure that runs evaluate_record on the record and condition
        Rc::new(move |row: &Record| {
            let Some(res) = cond.evaluate_record(row) else {
                return false;
            };
            res
        })
    } else {
        // closure the just returns true
        Rc::new(|_row: &Record| true)
    }
}

#[derive(Debug, PartialEq)]
pub struct BasicSelectStatementInner<'a> {
    pub columns: Vec<String>,
    pub table: &'a str,
    pub condition: Option<Condition>,
}

#[derive(Debug, PartialEq)]
pub struct BasicSelectStatement {
    pub columns: Vec<String>,
    pub table: String,
    pub condition: Option<Condition>,
}

impl BasicSelectStatement {
    pub fn new(sql_command: &str) -> Result<BasicSelectStatement> {
        let parser = select_with_where();
        let Ok((_, inner_sql_tree)) = parser.parse(sql_command) else {
            bail!("Invalid sql statement")
        };

        let BasicSelectStatementInner {
            columns: columns_inner,
            table: table_inner,
            condition,
        } = inner_sql_tree;

        let columns = columns_inner
            .into_iter()
            .map(|s| s.to_string().to_uppercase())
            .collect::<Vec<String>>();

        let table = table_inner.to_string().to_uppercase();

        Ok(BasicSelectStatement {
            columns,
            table,
            condition,
        })
    }

    pub fn condition_columns(&self) -> Option<Vec<String>> {
        if let Some(ref cond) = self.condition {
            let cond_cols = cond.columns_evaluated_in_condition();
            Some(cond_cols)
        } else {
            None
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Condition {
    Column(String),
    Value(Value),
    Operation {
        left: Box<Condition>,
        op: Operator,
        right: Box<Condition>,
    },
}

impl Condition {
    pub fn columns_evaluated_in_condition(&self) -> Vec<String> {
        // recursivly walk the condition tree and fetch all the columns needed for the query
        let mut columns: Vec<String> = Vec::new();
        match self {
            Self::Operation { left, right, .. } => {
                columns.extend(left.columns_evaluated_in_condition());
                columns.extend(right.columns_evaluated_in_condition())
            }
            Self::Column(v) => columns.push(v.to_uppercase()),
            _ => {}
        }
        columns
    }

    pub fn evaluate_record(&self, row: &Record) -> Option<bool> {
        // if left and right are Column/Value then evalute the condition
        // we should not get down to a Column/Value
        match self {
            Self::Operation { left, op, right } => match op {
                Operator::Or => Some(left.evaluate_record(row)? || right.evaluate_record(row)?),
                Operator::And => Some(left.evaluate_record(row)? && right.evaluate_record(row)?),
                _ => {
                    let left_v = left.resolve(row)?;
                    let right_v = right.resolve(row)?;
                    let result = Self::binary_op(left_v, op.to_owned(), right_v);
                    Some(result)
                }
            },
            _ => panic!("Invalid operation evaluation"),
        }
    }

    fn resolve(&self, row: &Record) -> Option<Value> {
        match self {
            Self::Column(c) => row.row_values.get(c).cloned(),
            Self::Value(v) => Some(v.clone()),
            Self::Operation { .. } => Some(Value::Bool(self.evaluate_record(row)?)),
        }
    }

    fn binary_op(lhs: Value, op: Operator, rhs: Value) -> bool {
        match (lhs, rhs) {
            (Value::Null, Value::Null) => true,
            (Value::Int(l), Value::Int(r)) => match op {
                Operator::Equal => l == r,
                Operator::NotEqual => l != r,
                Operator::LessThan => l < r,
                Operator::LessThanOrEqualTo => l <= r,
                Operator::GreaterThan => l > r,
                Operator::GreaterThanOrEqualTo => l >= r,
                _ => false,
            },
            (Value::Float(l), Value::Float(r)) => match op {
                Operator::Equal => l == r,
                Operator::NotEqual => l != r,
                Operator::LessThan => l < r,
                Operator::LessThanOrEqualTo => l <= r,
                Operator::GreaterThan => l > r,
                Operator::GreaterThanOrEqualTo => l >= r,
                _ => false,
            },
            (Value::String(l), Value::String(r)) => match op {
                Operator::Equal => l == r,
                Operator::NotEqual => l != r,
                _ => false,
            },
            _ => false,
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Operator {
    And,
    Or,
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    LessThanOrEqualTo,
    GreaterThanOrEqualTo,
}

// this is returned from a parser function
// it returns the part that has been parsed out and then the remaining portion of the text/tokens
type ParseResult<'a, Output> = Result<(&'a str, Output), String>;

// a trait with the function parse which takes in a str and returns some parsed str and additional
// tokens in output
pub trait Parser<'a, Output> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, Output>;
}

// a container that holds a function pointer to a boxed parser function
pub struct BoxedParser<'a, Output> {
    f: Box<dyn Fn(&'a str) -> ParseResult<'a, Output> + 'a>,
}

// accept a parser function and box it
impl<'a, Output> BoxedParser<'a, Output> {
    fn new<F>(f: F) -> BoxedParser<'a, Output>
    where
        F: Fn(&'a str) -> ParseResult<'a, Output> + 'a,
    {
        BoxedParser { f: Box::new(f) }
    }
}

// take the boxed parser function and parse the input
impl<'a, Output> Parser<'a, Output> for BoxedParser<'a, Output> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, Output> {
        // need to wrap f in () because it is a field with a function pointer and not a method on
        // the struct
        // in order to just for self.f(), f needs to be a method on the struct
        // in this case it is a function pointer in a field and not a function
        (self.f)(input)
    }
}

pub trait ParserExt<'a, Output>: Parser<'a, Output> + Sized {
    fn then<P2, Output2>(self, next: P2) -> BoxedParser<'a, (Output, Output2)>
    where
        P2: Parser<'a, Output2> + 'a,
        Output2: 'a,
        Output: 'a,
        Self: 'a,
    {
        BoxedParser::new(move |input| {
            self.parse(input).and_then(|(next_input, result)| {
                next.parse(next_input)
                    .map(|(final_input, next_result)| (final_input, (result, next_result)))
            })
        })
    }

    fn or<OtherParser>(self, other: OtherParser) -> BoxedParser<'a, Output>
    where
        OtherParser: Parser<'a, Output> + 'a,
        Self: 'a,
    {
        let p1 = Rc::new(self);
        let p2 = Rc::new(other);

        BoxedParser::new(move |input| match p1.parse(input) {
            Ok(res) => Ok(res),
            Err(_) => p2.parse(input),
        })
    }

    fn map<F, Output2>(self, f: F) -> BoxedParser<'a, Output2>
    where
        F: Fn(Output) -> Output2 + 'a,
        Output: 'a,
        Output2: 'a,
        Self: 'a,
    {
        BoxedParser::new(move |input| self.parse(input).map(|(rest, val)| (rest, f(val))))
    }
}

impl<'a, Output, P> ParserExt<'a, Output> for P where P: Parser<'a, Output> + Sized {}

// then define functions to return a BoxedParser
// define a closure then move that closure into a boxed parser
#[allow(dead_code)]
fn digit_char<'a>() -> impl Parser<'a, char> {
    BoxedParser::new(|input| {
        let mut chars = input.chars();
        if let Some(first) = chars.next() {
            if first.is_ascii_digit() {
                let rest = &input[first.len_utf8()..];
                return Ok((rest, first));
            }
        }
        Err("Expected digit".into())
    })
}

#[allow(dead_code)]
fn multi_digit_char<'a, Output, P>(parser: P) -> impl Parser<'a, Vec<Output>>
where
    P: Parser<'a, Output> + 'a,
    Output: 'a,
{
    BoxedParser::new(move |mut input| {
        let mut res: Vec<Output> = Vec::new();

        while let Ok((next, val)) = parser.parse(input) {
            res.push(val);
            input = next;
        }
        Ok((input, res))
    })
}

fn parse_for_select<'a>() -> impl Parser<'a, &'a str> {
    BoxedParser::new(|input| {
        let keyword = "SELECT";

        let (head, rest) = input.split_at(keyword.len());

        if head.eq_ignore_ascii_case(head) {
            Ok((rest, keyword))
        } else {
            Err("Invalid query syntax".into())
        }
    })
}

fn _parse_for_where<'a>() -> impl Parser<'a, &'a str> {
    BoxedParser::new(|input| {
        let keyword = "WHERE";

        let (head, rest) = input.split_at(keyword.len());

        if head.eq_ignore_ascii_case(keyword) {
            Ok((rest, keyword))
        } else {
            Err("Invalid query syntax".into())
        }
    })
}

fn parse_for_int<'a>() -> impl Parser<'a, Condition> {
    BoxedParser::new(|input| {
        let mut i = 0_usize;
        for c in input.chars() {
            if c.is_numeric() {
                i += 1;
            } else {
                break;
            }
        }

        let value = &input[..i].parse::<i64>();
        match value {
            Ok(v) => Ok((&input[i..], Condition::Value(Value::Int(*v)))),
            Err(_) => Err("Not an integer".into()),
        }
    })
}

fn parse_for_float<'a>() -> impl Parser<'a, Condition> {
    // first parse out the whole number
    // then check for a .
    // then parse until not a number
    BoxedParser::new(|input| {
        let mut float_buffer = String::new();
        let mut chars = input.chars().peekable();
        while let Some(c) = chars.next_if(|&c| c.is_numeric()) {
            float_buffer.push(c);
        }

        let Some(decimal) = chars.next_if(|&c| c == '.') else {
            return Err("Not an float".into());
        };
        float_buffer.push(decimal);

        while let Some(c) = chars.next_if(|&c| c.is_numeric()) {
            float_buffer.push(c);
        }

        let value = float_buffer.parse::<f64>();
        let i = chars.collect::<Vec<_>>().len();
        match value {
            Ok(v) => Ok((&input[i..], Condition::Value(Value::Float(v)))),
            Err(_) => Err("Not an integer".into()),
        }
    })
}

fn parse_for_string_value<'a>() -> impl Parser<'a, Condition> {
    BoxedParser::new(|input| {
        let Some(first) = input.chars().nth(0) else {
            return Err("Equality value not a string".into());
        };

        if first != '\'' {
            return Err("Equality value not a string".into());
        }
        let Some(mut s) = input.find('\'') else {
            return Err("equality value not a string".into());
        };
        s += 1;
        let Some(e) = input[s..].find('\'') else {
            return Err("equality value not a string".into());
        };

        Ok((
            &input[s + e + 1..],
            Condition::Value(Value::String(input[s..s + e].to_string())),
        ))
    })
}

fn parse_for_condition_column<'a>() -> impl Parser<'a, Condition> {
    BoxedParser::new(|input| {
        let identifier_parser = parse_for_identifiers();

        let Ok((next, res)) = identifier_parser.parse(input) else {
            return Err("not an identifer".into());
        };

        Ok((next, Condition::Column(res.to_uppercase())))
    })
}

fn parse_for_from<'a>() -> impl Parser<'a, &'a str> {
    BoxedParser::new(|input| {
        let keyword = "FROM";

        let (head, rest) = input.split_at(keyword.len());

        if head.eq_ignore_ascii_case(keyword) {
            Ok((rest, keyword))
        } else {
            Err("Invalid query syntax".into())
        }
    })
}

// parse until we reach then end of a word
fn parse_for_identifiers<'a>() -> impl Parser<'a, &'a str> {
    BoxedParser::new(|input| {
        let mut end = 0_usize;

        for c in input.chars() {
            if c.is_ascii_alphanumeric() || c == '_' {
                end += 1
            } else {
                break;
            }
        }

        if end == 0 {
            Err("expected identifier".into())
        } else {
            Ok((&input[end..], &input[..end]))
        }
    })
}

// parse until we reach a space
// then return the remaining input and an empty ()
fn parse_for_whitespace<'a>() -> impl Parser<'a, ()> {
    BoxedParser::new(|input| {
        let len = input.len();
        let mut end = 0_usize;
        for c in input.bytes() {
            if matches!(c, 32 | 9 | 10 | 11 | 12 | 13) {
                end += 1
            } else {
                break;
            }
        }

        if end == len {
            Ok((input, ()))
        } else {
            Ok((&input[end..], ()))
        }
    })
}

fn parse_for_operator<'a>() -> impl Parser<'a, Operator> {
    BoxedParser::new(|input| {
        let mut chars = input.chars().peekable();
        let Some(first) = chars.next() else {
            return Err("String is empty".into());
        };
        match first {
            '=' => {
                // return Operator::Equals
                return Ok((&input[1..], Operator::Equal));
            }
            '!' => {
                // check to make sure that the next character is =
                let Some(next) = chars.next() else {
                    return Err("Invalid operator".into());
                };
                if !matches!(next, '=') {
                    return Err("Invalid operator".into());
                }
                return Ok((&input[2..], Operator::NotEqual));
            }
            '<' => {
                // check to make sure that the next character is =
                let Some(next) = chars.next() else {
                    return Ok((&input[1..], Operator::LessThan));
                };
                if !matches!(next, '=') {
                    return Ok((&input[1..], Operator::LessThan));
                }
                return Ok((&input[2..], Operator::LessThanOrEqualTo));
            }
            '>' => {
                // check to make sure that the next character is =
                let Some(next) = chars.next() else {
                    return Ok((&input[1..], Operator::GreaterThan));
                };
                if !matches!(next, '=') {
                    return Ok((&input[1..], Operator::GreaterThan));
                }
                return Ok((&input[2..], Operator::GreaterThanOrEqualTo));
            }
            _ => return Err("Invalid operator".into()),
        }
    })
}
/*
* Start at or predence level
*   - parse for OR and split
* Pass each new blob down to the AND precednece level
*   - parse for AND and split
* parse for binary operators
* then construct the AST from the bottom up
* */

fn fold_conditions(op: Operator, conditions: Vec<Condition>) -> Option<Condition> {
    let mut cond_iter = conditions.into_iter();

    let first = cond_iter.next()?;

    Some(cond_iter.fold(first, |left, right| Condition::Operation {
        left: Box::new(left),
        op,
        right: Box::new(right),
    }))
}

fn or_precedence<'a>() -> impl Parser<'a, Option<Condition>> {
    let and_parser = and_precedence();
    BoxedParser::new(move |input| {
        let input_len = input.len();
        let mut start = 0_usize;
        let mut end = input.to_uppercase().find(" OR ").unwrap_or(input_len);
        let mut ops = Vec::<Condition>::new();
        while start < input_len {
            let Ok((_, op)) = and_parser.parse(&input[start..end]) else {
                return Err("Invalid where condition or".into());
            };
            ops.push(op);
            start = std::cmp::min(end + 4, input_len);
            end = input.to_lowercase().find(" OR ").unwrap_or(input_len);
        }

        let folded_op = fold_conditions(Operator::Or, ops);
        Ok(("", folded_op))
    })
}

fn and_precedence<'a>() -> impl Parser<'a, Condition> {
    let binary_op_parser = binary_op_precedence();
    BoxedParser::new(move |input| {
        let input_len = input.len();
        let mut start = 0_usize;
        let mut end = input.to_uppercase().find(" AND ").unwrap_or(input_len);
        let mut ops = Vec::<Condition>::new();
        while start < input_len {
            let Ok((_, op)) = binary_op_parser.parse(&input[start..end]) else {
                return Err("Invalid where condition and".into());
            };
            ops.push(op);
            start = std::cmp::min(end + 5, input_len);
            end = input.to_lowercase().find(" AND ").unwrap_or(input_len);
        }

        let Some(folded_op) = fold_conditions(Operator::And, ops) else {
            return Err("No ops to fold".into());
        };
        Ok(("", folded_op))
    })
}

fn binary_op_precedence<'a>() -> impl Parser<'a, Condition> {
    // parse out left
    // parse out whitespace
    // parse out operator
    // parse out right
    let whitespace = parse_for_whitespace();

    BoxedParser::new(move |mut input| {
        let value_parser = parse_for_float()
            .or(parse_for_int())
            .or(parse_for_string_value())
            .or(parse_for_condition_column());

        let op_parser = parse_for_operator();
        if let Ok((next, _)) = whitespace.parse(input) {
            input = next;
        }
        let Ok((mut input, left)) = value_parser.parse(input) else {
            return Err("no left side value".into());
        };

        if let Ok((next, _)) = whitespace.parse(input) {
            input = next;
        }

        let Ok((mut input, op)) = op_parser.parse(input) else {
            return Err("no operator".into());
        };
        if let Ok((next, _)) = whitespace.parse(input) {
            input = next;
        }

        let Ok((input, right)) = value_parser.parse(input) else {
            return Err("no right side value".into());
        };
        Ok((
            input,
            Condition::Operation {
                left: Box::new(left),
                op,
                right: Box::new(right),
            },
        ))
    })
}

fn parse_for_condition<'a>() -> impl Parser<'a, Option<Condition>> {
    or_precedence()
}

fn sep_by_two_seperators<'a, P, S, W, Output>(
    item: P,
    whitespace: W, // seperator on whitespace
    seperator: S,  // seperator on comma
) -> impl Parser<'a, Vec<Output>>
where
    P: Parser<'a, Output> + 'a,
    S: Parser<'a, ()> + 'a,
    W: Parser<'a, ()> + 'a,
    Output: 'a,
{
    BoxedParser::new(move |mut input| {
        let mut res_items = Vec::new();

        // parse out the first identifier from the input
        match item.parse(input) {
            Ok((next, res)) => {
                res_items.push(res);
                input = next;
            }
            Err(_) => return Ok((input, res_items)),
        }

        // parse for seperator
        // then parse for whitespace
        // until it is exhausted
        while let Ok((next, _)) = seperator.parse(input) {
            input = next;

            match whitespace.parse(input) {
                Ok((next, _)) => {
                    input = next;
                }
                Err(_) => {}
            };
            match item.parse(input) {
                Ok((next, res)) => {
                    res_items.push(res);
                    input = next;
                }
                Err(e) => return Err(e),
            }
        }
        Ok((input, res_items))
    })
}

fn parse_for_keyword<'a>(keyword: &'a str) -> impl Parser<'a, ()> {
    BoxedParser::new(move |input| {
        let klen = keyword.len();
        if input.len() < klen {
            return Err(format!("expected {}", keyword));
        };
        let (head, rest) = input.split_at(keyword.len());

        if head.eq_ignore_ascii_case(keyword) {
            Ok((rest, ()))
        } else {
            Err(format!("expected {}", keyword))
        }
    })
}

fn _parse_for_create<'a>() -> impl Parser<'a, ()> {
    parse_for_keyword("CREATE")
}

fn _parse_for_table<'a>() -> impl Parser<'a, ()> {
    BoxedParser::new(|input| {
        let target = "TABLE";

        let (head, rest) = input.split_at(target.len());

        if head.eq_ignore_ascii_case(target) {
            Ok((rest, ()))
        } else {
            Err("expected TABLE".into())
        }
    })
}

fn parse_for_schema_multi_word_column_name<'a>() -> impl Parser<'a, &'a str> {
    BoxedParser::new(move |input| {
        let Some(mut start) = input.find('"') else {
            return Err("Not a multi word column name".into());
        };

        start = start.saturating_add(1);

        let Some(mut end) = input[start..].find('"') else {
            return Err("Not a multi word column name".into());
        };

        // index passed "
        let new_start = end.saturating_add(2);
        end += 1;

        Ok((&input[new_start..], &input[start..end]))
    })
}

fn parse_for_column_constraints<'a>() -> impl Parser<'a, Vec<SQLiteColumnConstraints>> {
    // TODO: expand this to all possible constraint possiblities
    BoxedParser::new(|input| {
        let mut token_stream = TokenStream::new(input);
        if token_stream.len() == 0 {
            return Ok(("", Vec::new()));
        }

        let mut res: Vec<SQLiteColumnConstraints> = Vec::new();
        while let Some(token) = token_stream.next() {
            match token {
                SqlConditionToken::Keyword(k) => match k {
                    SQLiteKeyword::Primary => {
                        if matches!(
                            token_stream.peek(),
                            Some(&SqlConditionToken::Keyword(SQLiteKeyword::Key))
                        ) {
                            res.push(SQLiteColumnConstraints::PrimaryKey);
                            let _ = token_stream.next();
                        } else {
                            return Err("Invalid conditon".into());
                        }
                    }
                    SQLiteKeyword::Not => {
                        if matches!(
                            token_stream.peek(),
                            Some(&SqlConditionToken::Keyword(SQLiteKeyword::Null))
                        ) {
                            res.push(SQLiteColumnConstraints::NotNull);
                            let _ = token_stream.next();
                        } else {
                            return Err("Invalid conditon".into());
                        }
                    }
                    SQLiteKeyword::AutoIncrement => {
                        res.push(SQLiteColumnConstraints::AutoIncrement)
                    }
                    SQLiteKeyword::Collate => {
                        let Some(collation_condition) = token_stream.next() else {
                            return Err("Invalid collalition".into());
                        };

                        let c_str = match collation_condition {
                            SqlConditionToken::StringGroup(v) => v,
                            SqlConditionToken::Identifier(v) => v,
                            SqlConditionToken::QuotedIdentifier(v) => v,
                            _ => return Err("Invalid collation".into()),
                        };
                        res.push(SQLiteColumnConstraints::Collate(c_str.to_string()));
                    }
                    SQLiteKeyword::Ascending => res.push(SQLiteColumnConstraints::Ascending),
                    SQLiteKeyword::Descending => res.push(SQLiteColumnConstraints::Descending),
                    _ => todo!(),
                },
                _ => todo!(),
            }
        }

        Ok(("", res))
    })
}

fn parse_for_string_group<'a>() -> impl Parser<'a, &'a str> {
    BoxedParser::new(|input| {
        let mut chars = input.chars();

        let Some(first) = chars.next() else {
            return Err("Invalid string group".into());
        };

        if first != '"' {
            return Err("Invalid string group".into());
        }

        let start = 1_usize;
        let mut end = start;

        for c in chars {
            if c == '"' {
                break;
            }
            end += 1;
        }

        Ok((&input[end + 1..], &input[start..end]))
    })
}

fn parse_column_type<'a>() -> impl Parser<'a, Option<SQLiteColumnType>> {
    let parser = parse_for_identifiers();
    BoxedParser::new(move |input| {
        if input.is_empty() {
            return Ok(("", None));
        };
        let Ok((rest, type_str)) = parser.parse(input) else {
            return Err("No identifier found".into());
        };

        let Ok(dtype) = SQLiteColumnType::try_from(type_str) else {
            return Err("Invalid type".into());
        };

        Ok((rest, Some(dtype)))
    })
}

fn parse_table_column_definition<'a>() -> impl Parser<'a, TableColumn> {
    parse_for_whitespace() // white space
        .then(parse_for_schema_multi_word_column_name().or(parse_for_identifiers())) // column name
        .then(parse_for_whitespace())
        .then(parse_column_type()) // type
        .then(parse_for_column_constraints())
        .map(|(((((), name), _), data_type), constraints)| TableColumn {
            name: name.to_string(),
            data_type,
            constraints,
        })
}

fn _parse_for_open_paren<'a>() -> impl Parser<'a, ()> {
    BoxedParser::new(|input| {
        let Some(start) = input.find('(') else {
            return Err("No open paren found".into());
        };

        Ok((&input[start + 1..], ()))
    })
}

fn _parse_for_closed_paren<'a>() -> impl Parser<'a, ()> {
    BoxedParser::new(|input| {
        let Some(start) = input.find(')') else {
            return Err("No open paren found".into());
        };

        Ok((&input[start + 1..], ()))
    })
}

fn parse_for_column_schema_group<'a>() -> impl Parser<'a, &'a str> {
    BoxedParser::new(|input| {
        let Some(mut start) = input.find('(') else {
            return Err("Invalid schema".into());
        };
        start += 1;
        let mut end = start;
        let mut paren_depth = 1_usize;
        let mut chars = input.chars();
        let _ = chars.next();

        // grab the inner token group
        while let Some(next) = chars.next() {
            match next {
                '(' => paren_depth += 1,
                ')' => paren_depth -= 1,
                _ => {}
            }
            end += 1;
            if paren_depth == 0 {
                break;
            }
        }
        Ok((&input[end..], &input[start..end - 1]))
    })
}

fn parse_for_table_schema_columns<'a>() -> impl Parser<'a, Vec<TableColumn>> {
    let col_parser = parse_table_column_definition();
    let group_parser = parse_for_column_schema_group();
    BoxedParser::new(move |mut input| {
        let mut columns: Vec<TableColumn> = Vec::new();

        let Ok((rest, col_tokens)) = group_parser.parse(input) else {
            return Err("Invalid schema definition".into());
        };

        input = rest;

        let mut group_depth = GroupDepth::new();
        let mut line_start = 0_usize;
        let mut line_end = 0_usize;
        for token in col_tokens.chars() {
            if group_depth.depth == 0 && token == ',' {
                let Ok((_, column)) = col_parser.parse(&col_tokens[line_start..line_end]) else {
                    return Err("Invalid colum definition".into());
                };
                columns.push(column);
                line_start = line_end + 1;
                line_end = line_start;
            } else {
                group_depth.update(token);
                line_end += 1;
            }
        }

        let Ok((_, column)) = col_parser.parse(&col_tokens[line_start..line_end]) else {
            return Err("Invalid colum definition".into());
        };

        columns.push(column);

        Ok((input, columns))
    })
}

fn _parse_for_without_row_id<'a>() -> impl Parser<'a, bool> {
    BoxedParser::new(|input| {
        let token = "WITHOUT ROWID";
        let token_len = token.len();
        if input.len() < token_len {
            return Ok((input, false));
        }
        let (head, rest) = input.split_at(token.len());

        if head.eq_ignore_ascii_case(token) {
            Ok((&rest, true))
        } else {
            Ok((&rest, false))
        }
    })
}

fn parse_for_optional_keyword<'a>(keyword: &'a str) -> impl Parser<'a, bool> {
    BoxedParser::new(move |input| {
        let keyword_len = keyword.len();
        if input.len() < keyword_len {
            return Ok((input, false));
        }
        let (head, rest) = input.split_at(keyword_len);

        if head.eq_ignore_ascii_case(keyword) {
            Ok((&rest, true))
        } else {
            Ok((input, false))
        }
    })
}

fn if_not_exists_parser<'a>() -> impl Parser<'a, bool> {
    BoxedParser::new(|input| {
        let keyword = "IF NOT EXISTS";
        let len = keyword.len();
        if input.len() < len {
            return Ok((input, false));
        }

        let (head, rest) = input.split_at(len);

        if head.eq_ignore_ascii_case(keyword) {
            Ok((rest, true))
        } else {
            Ok((input, false))
        }
    })
}

fn parse_index_column_definition<'a>() -> impl Parser<'a, IndexColumn> {
    parse_for_whitespace() // white space
        .then(parse_for_schema_multi_word_column_name().or(parse_for_identifiers()))
        .then(parse_for_whitespace())
        .then(parse_for_column_constraints())
        .map(|((((), name), _), constraints)| IndexColumn {
            name: name.to_string(),
            constraints,
        })
}

fn parse_index_columns<'a>() -> impl Parser<'a, Vec<IndexColumn>> {
    let paren_group_parser = parse_for_column_schema_group();
    let col_parser = parse_index_column_definition();
    BoxedParser::new(move |input| {
        let mut index_columns: Vec<IndexColumn> = Vec::new();
        let Ok((rest, col_tokens)) = paren_group_parser.parse(input) else {
            return Err("Invalid schema".into());
        };

        let mut group_depth = GroupDepth::new();
        let mut line_start = 0_usize;
        let mut line_end = 0_usize;

        for token in col_tokens.chars() {
            if group_depth.depth == 0 && token == ',' {
                let Ok((_, column)) = col_parser.parse(&col_tokens[line_start..line_end]) else {
                    return Err("Invalid colum definition".into());
                };
                index_columns.push(column);
                line_start = line_end + 1;
                line_end = line_start;
            } else {
                group_depth.update(token);
                line_end += 1;
            }
        }
        let Ok((_, column)) = col_parser.parse(&col_tokens[line_start..line_end]) else {
            return Err("Invalid colum definition".into());
        };

        index_columns.push(column);

        Ok((rest, index_columns))
    })
}

// TODO: add a parser for an index schema
pub fn parse_index_schema<'a>() -> impl Parser<'a, (String, String, bool, Vec<IndexColumn>)> {
    // CREATE
    // INDEX
    // optional IF NOT EXISTS
    // index name
    // ON
    // base table name
    // paren group
    //      column name <Constraints>
    parse_for_whitespace()
        .then(parse_for_keyword("CREATE"))
        .then(parse_for_whitespace())
        .then(parse_for_optional_keyword("UNIQUE"))
        .then(parse_for_whitespace())
        .then(parse_for_keyword("INDEX"))
        .then(parse_for_whitespace())
        .then(if_not_exists_parser())
        .then(parse_for_whitespace())
        .then(parse_for_string_group().or(parse_for_identifiers()))
        .then(parse_for_whitespace())
        .then(parse_for_keyword("ON"))
        .then(parse_for_whitespace())
        .then(parse_for_string_group().or(parse_for_identifiers()))
        .then(parse_for_whitespace())
        .then(parse_index_columns())
        .map(
            |(
                (
                    (
                        (
                            ((((((((((((), _), _), unique), _), _), _), index_name), _), _), _), _),
                            _,
                        ),
                        base_table_name,
                    ),
                    _,
                ),
                index_columns,
            )| {
                (
                    index_name.to_string(),
                    base_table_name.to_string(),
                    unique,
                    index_columns,
                )
            },
        )
}

pub fn parse_table_schema<'a>() -> impl Parser<'a, (String, Vec<TableColumn>, PrimaryKey, bool)> {
    parse_for_whitespace()
        .then(parse_for_keyword("CREATE"))
        .then(parse_for_whitespace())
        .then(parse_for_keyword("TABLE"))
        .then(parse_for_whitespace())
        .then(parse_for_string_group().or(parse_for_identifiers()))
        .then(parse_for_whitespace())
        .then(parse_for_table_schema_columns())
        .then(parse_for_whitespace())
        .then(parse_for_optional_keyword("WITHOUT ROWID"))
        .map(
            |((((((((_, _), _), _), name), _), columns), _), without_rowid)| {
                // parse for the primary key
                let primary_key = PrimaryKey::from_table_columns(&columns);
                (name.to_string(), columns, primary_key, without_rowid)
            },
        )
}

fn parse_for_columns<'a>() -> impl Parser<'a, Vec<&'a str>> {
    sep_by_two_seperators(
        parse_for_identifiers(),
        parse_for_whitespace(),
        BoxedParser::new(|input| {
            if input.starts_with(",") {
                return Ok((&input[1..], ()));
            } else {
                return Err("Expected comma".into());
            }
        }),
    )
}

pub fn basic_select_statement<'a>() -> impl Parser<'a, BasicSelectStatementInner<'a>> {
    parse_for_select()
        .then(parse_for_whitespace())
        .then(parse_for_columns())
        .then(parse_for_whitespace())
        .then(parse_for_from())
        .then(parse_for_whitespace())
        .then(parse_for_identifiers())
        .map(|((((((_, _), base_cols), _), _), _), table)| {
            let columns: Vec<String> = base_cols.into_iter().map(|c| c.to_uppercase()).collect();
            BasicSelectStatementInner {
                columns,
                table,
                condition: None,
            }
        })
}

fn parse_for_where_clause<'a>() -> impl Parser<'a, Option<Condition>> {
    parse_for_whitespace()
        .then(parse_for_keyword("WHERE"))
        .then(parse_for_condition())
        .map(|((_, _), condition)| condition)
        .or(BoxedParser::new(|input| Ok((input, None))))
}

pub fn select_with_where<'a>() -> impl Parser<'a, BasicSelectStatementInner<'a>> {
    parse_for_keyword("SELECT")
        .then(parse_for_whitespace())
        .then(parse_for_columns())
        .then(parse_for_whitespace())
        .then(parse_for_keyword("FROM"))
        .then(parse_for_whitespace())
        .then(parse_for_identifiers())
        .then(parse_for_where_clause())
        .map(|(((((((_, _), base_cols), _), _), _), table), condition)| {
            let columns: Vec<String> = base_cols.into_iter().map(|c| c.to_uppercase()).collect();
            BasicSelectStatementInner {
                columns,
                table,
                condition,
            }
        })
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_single() {
        let parser = digit_char();
        let res = parser.parse("1122aa");
        assert_eq!(res, Ok(("122aa", '1')))
    }

    #[test]
    fn test_multiple() {
        let digit_parser = digit_char();
        let mdp = multi_digit_char(digit_parser);
        let res = mdp.parse("11122aa");
        assert_eq!(res, Ok(("aa", vec!['1', '1', '1', '2', '2'])))
    }

    #[test]
    fn select_parser() {
        let q = "Select * from table";
        let parser = parse_for_select();
        let res = parser.parse(q).unwrap();
        assert_eq!(res, (" * from table", "SELECT"))
    }

    #[test]
    fn parser_for_identifier_and_whitespace() {
        let parser = parse_for_identifiers();
        let res = parser.parse("word1 word2").unwrap();

        let wparser = parse_for_whitespace();

        let res2 = wparser.parse(res.0).unwrap();
        assert_eq!(res2, ("word2", ()))
    }

    #[test]
    fn test_chain() {
        let chain = parse_for_select()
            .then(parse_for_whitespace())
            .then(parse_for_columns())
            .then(parse_for_whitespace())
            .then(parse_for_from())
            .then(parse_for_whitespace())
            .then(parse_for_identifiers())
            .map(|input| println!("input: {:?}", input));
        let _ = chain.parse("select column from table");
    }

    #[test]
    fn test_whole_parser() {
        let parser = basic_select_statement();
        let res = parser.parse("select column from table").unwrap();
        let actual = BasicSelectStatementInner {
            columns: ["column".into()].to_vec(),
            table: "table",
            condition: None,
        };
        assert_eq!(res, ("", actual))
    }

    #[test]
    fn multiple_columns() {
        let parser = basic_select_statement();
        let res = parser.parse("select column1, column2 from table").unwrap();
        let actual = BasicSelectStatementInner {
            columns: ["column1".into(), "column2".into()].to_vec(),
            table: "table",
            condition: None,
        };
        assert_eq!(res, ("", actual))
    }

    #[test]
    fn float_parser() {
        let parser = parse_for_float();
        let result = parser.parse("12.0 abc");
        let res = result.unwrap();
        assert_eq!(res, (" abc", Condition::Value(Value::Float(12.0))))
    }

    #[test]
    fn int_parser() {
        let parser = parse_for_int();
        let result = parser.parse("12 abc");
        let res = result.unwrap();
        assert_eq!(res, (" abc", Condition::Value(Value::Int(12))))
    }

    #[test]
    fn string_value_parser() {
        let parser = parse_for_string_value();
        let result = parser.parse("'Yellow' ");
        let res = result.unwrap();
        assert_eq!(
            res,
            (" ", Condition::Value(Value::String(String::from("Yellow"))))
        )
    }

    #[test]
    fn test_parse_for_operator_geoet() {
        let parser = parse_for_operator();
        let res = parser.parse(">= thing").unwrap();
        assert_eq!(res, (" thing", Operator::GreaterThanOrEqualTo));
    }

    #[test]
    fn test_parse_for_operator_leoet() {
        let parser = parse_for_operator();
        let res = parser.parse("<= thing").unwrap();
        assert_eq!(res, (" thing", Operator::LessThanOrEqualTo));
    }

    #[test]
    fn test_parse_for_operator_ne() {
        let parser = parse_for_operator();
        let res = parser.parse("!= thing").unwrap();
        assert_eq!(res, (" thing", Operator::NotEqual));
    }

    #[test]
    fn test_parse_for_operator_gt() {
        let parser = parse_for_operator();
        let res = parser.parse("> thing").unwrap();
        assert_eq!(res, (" thing", Operator::GreaterThan));
    }

    #[test]
    fn test_parse_for_operator_lt() {
        let parser = parse_for_operator();
        let res = parser.parse("< thing").unwrap();
        assert_eq!(res, (" thing", Operator::LessThan));
    }

    #[test]
    fn test_parse_for_operator_e() {
        let parser = parse_for_operator();
        let res = parser.parse("= thing").unwrap();
        assert_eq!(res, (" thing", Operator::Equal));
    }

    #[test]
    fn test_parse_for_condition_identifer() {
        let parser = parse_for_condition_column();
        let res = parser.parse("thing = ").unwrap();
        assert_eq!(res, (" = ", Condition::Column(String::from("THING"))));
    }

    #[test]
    fn test_basic_condition() {
        let condition_parser = parse_for_condition();
        let res = condition_parser.parse("value >= 42").unwrap();
        assert_eq!(
            res,
            (
                "",
                Some(Condition::Operation {
                    left: Box::new(Condition::Column(String::from("VALUE"))),
                    op: Operator::GreaterThanOrEqualTo,
                    right: Box::new(Condition::Value(Value::Int(42)))
                })
            )
        )
    }

    #[test]
    fn test_condition_w_and() {
        let condition_parser = parse_for_condition();
        let res = condition_parser
            .parse("value >= 42 and value2 = 1")
            .unwrap();
        println!("with and statement {:?}", res);
    }

    #[test]
    fn test_select_with_where() {
        let parser = select_with_where();
        let res = parser
            .parse("select value, value2 from column where value < 42 and value2 > 30.0")
            .unwrap();
        println!("full select statement: {:?}", res);
    }

    #[test]
    fn test_condition_eval() {
        let condition = Condition::Operation {
            left: Box::new(Condition::Column(String::from("value"))),
            op: Operator::LessThan,
            right: Box::new(Condition::Value(Value::Int(42_i64))),
        };

        let mut row_values: std::collections::HashMap<String, Value> =
            std::collections::HashMap::new();
        row_values.insert(String::from("value"), Value::Int(50));
        let record = Record {
            record_id: 0,
            row_values,
        };

        let is_met = condition.evaluate_record(&record);
        assert_eq!(is_met, Some(false));
    }

    #[test]
    fn parse_for_multi_word_column_name() {
        let parser = parse_for_schema_multi_word_column_name();

        let (_, res) = parser.parse("\"SIZE RANGE\"").unwrap();
        assert_eq!(res, String::from("SIZE RANGE"))
    }

    #[test]
    fn test_column_def() {
        let col_str = "id integer primary key autoincrement";

        let col_parser = parse_table_column_definition();

        let (_, col) = col_parser.parse(col_str).unwrap();
        println!("col def: {:?}", col);
    }

    #[test]
    fn test_condition_columns() {
        let query = "SELECT id, name FROM companies WHERE country = 'micronesia'";

        let parsed_query = BasicSelectStatement::new(query).unwrap();

        let cols = parsed_query.condition_columns().unwrap();

        assert_eq!(cols, vec!["COUNTRY"])
    }
}
